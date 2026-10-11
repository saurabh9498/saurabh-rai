"""Control-flow tests for the Builder + Judge pipeline, using a scripted model stand-in.

These prove the guardrails behave as designed (what gets rejected, retried, clarified or
executed). They do not measure how well any real model writes SQL; that is eval/'s job.
"""
from __future__ import annotations

import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

from nl2sql import judge  # noqa: E402
from nl2sql.confidence import score  # noqa: E402
from nl2sql.llm import Completion, ScriptedLLM  # noqa: E402
from nl2sql.pipeline import Pipeline, extract_sql  # noqa: E402
from nl2sql.rbac import ANALYST, MARKETING, violations  # noqa: E402
from nl2sql.schema import retrieve  # noqa: E402
from nl2sql.warehouse import execute, schema_only, with_data  # noqa: E402

SURE = [-0.01] * 20                     # every token ~99% likely
VAGUE = [-0.01] * 15 + [-2.5] * 5       # a few tokens at ~8%


def c(sql: str, lps=None) -> Completion:
    return Completion(f"```sql\n{sql}\n```", SURE if lps is None else lps)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    import pandas as pd
    from generate_events import generate

    d = tmp_path_factory.mktemp("wh")
    users, events = generate(2_000, 40_000, seed=1)
    users.to_parquet(d / "users.parquet", index=False)
    events.to_parquet(d / "events.parquet", index=False)
    seg = pd.DataFrame({"user_id": users.user_id, "segment": "casual", "spend_90d": 0.0, "views_30d": 0})
    seg.loc[:9, "segment"] = "high_value"
    seg.to_parquet(d / "segments.parquet", index=False)
    return d


def pipe(script, role=ANALYST, data_dir=None, semantic=None):
    return Pipeline.for_role(ScriptedLLM(script), role, data_dir, semantic_judge=semantic)


# --- happy path and retries -------------------------------------------------------------
def test_valid_query_executes(data_dir):
    p = pipe([c("SELECT COUNT(*) AS n FROM segments WHERE segment = 'high_value'")], data_dir=data_dir)
    r = p.ask("how many high value users are there")
    assert r.status == "ok" and r.rows == [(10,)] and r.attempts == 1


def test_judge_rejection_feeds_back_and_retry_succeeds():
    p = pipe([c("SELECT region, SUM(spend) FROM events GROUP BY region"),          # wrong column
              c("SELECT u.region, SUM(e.spend_usd) AS spend FROM events e JOIN users u USING (user_id) GROUP BY u.region")])
    r = p.ask("total spend by region")
    assert r.status == "ok" and r.attempts == 2
    assert "rejected" in p.builder.prompts[1] and "does not plan" in p.builder.prompts[1]


def test_gives_up_after_max_attempts():
    p = pipe([c("SELECT nope FROM events"), c("SELECT still_nope FROM events")])
    r = p.ask("events by nope")
    assert r.status == "rejected" and r.attempts == 2 and "does not plan" in r.reasons[0]


# --- confidence gating ----------------------------------------------------------------
def test_low_confidence_asks_instead_of_running():
    p = pipe([c("SELECT user_id FROM events ORDER BY spend_usd DESC LIMIT 10", VAGUE)])
    r = p.ask("top customers last month")
    assert r.status == "clarify" and r.clarifying_question and r.rows == []


def test_confidence_math():
    s = score([math.log(0.9), math.log(0.5)])
    assert s.geo_mean_prob == pytest.approx(math.sqrt(0.45))
    assert s.min_prob == pytest.approx(0.5)
    assert score([]).low(0.8, 0.2)


# --- safety gate -----------------------------------------------------------------------
@pytest.mark.parametrize("sql", [
    "DELETE FROM users",
    "DROP TABLE events",
    "COPY users TO 'out.csv'",
    "ATTACH 'other.db'",
    "SELECT 1; DROP TABLE users",
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM '/etc/passwd'",
])
def test_unsafe_sql_never_passes(sql):
    v = judge.check(sql, "q", ANALYST, schema_only())
    assert not v.ok


def test_locked_connection_cannot_read_files(data_dir):
    con = with_data(data_dir)
    with pytest.raises(Exception):
        execute(con, "SELECT * FROM read_csv('/etc/passwd')")
    with pytest.raises(Exception):
        con.execute("SET enable_external_access = true")


# --- RBAC --------------------------------------------------------------------------------
def test_marketing_cannot_read_restricted_columns_or_tables():
    assert violations("SELECT email FROM user_pii", MARKETING) == ["column user_pii.email", "table user_pii"]
    assert violations("SELECT signup_days_ago FROM users", MARKETING) == ["column users.signup_days_ago"]
    assert violations("SELECT * FROM users", MARKETING) == ["SELECT * on partially restricted table users"]
    assert violations("SELECT s.segment, u.region FROM segments s JOIN users u USING (user_id)", MARKETING) == []


def test_rbac_rejection_is_final_not_retried():
    p = pipe([c("SELECT email FROM user_pii"), c("SELECT 1")], role=ANALYST)
    r = p.ask("email addresses of lapsed users")
    assert r.status == "rejected" and r.attempts == 1 and "user_pii" in " ".join(r.reasons)


def test_prompt_only_shows_permitted_columns_and_count_star_is_allowed():
    p = pipe([c("SELECT region, COUNT(*) FROM users GROUP BY region")], role=MARKETING)
    r = p.ask("how many users per region and device")
    prompt = p.builder.prompts[0]
    assert "region" in prompt and "signup_days_ago" not in prompt and "email" not in prompt
    assert r.status == "ok", r.reasons


BYPASSES = [
    "SELECT SIGNUP_DAYS_AGO FROM users",
    'SELECT "SIGNUP_DAYS_AGO" FROM users',
    "SELECT user_id FROM users WHERE Signup_Days_Ago > 1400",
    "SELECT u FROM users u",
    "SELECT users FROM users",
    "SELECT to_json(u) FROM users u",
    "SELECT list(u) FROM users u",
    "SELECT COLUMNS('sign.*') FROM users",
    "SELECT #4 FROM users",
    "SELECT sd FROM users AS t(a, b, c, sd)",
    "SELECT * FROM main.user_pii",
    "WITH users AS (SELECT * FROM user_pii) SELECT * FROM users",
    "SELECT region FROM users UNION ALL SELECT email FROM user_pii",
]


@pytest.mark.parametrize("sql", BYPASSES)
def test_marketing_cannot_reach_restricted_data(sql, data_dir):
    # Layer 2: the Judge rejects it ...
    assert not judge.check(sql, "q", MARKETING, schema_only(MARKETING)).ok
    # ... and layer 3: even run directly on the role's connection, restricted data is absent.
    con = with_data(data_dir, MARKETING)
    try:
        cols, rows = execute(con, sql)
    except Exception:
        return                                              # failed to run: nothing leaked
    flat = repr(rows).lower() + repr(cols).lower()
    assert "signup_days_ago" not in flat and "@" not in flat


def test_role_connection_contains_only_permitted_columns(data_dir):
    con = with_data(data_dir, MARKETING)
    tables = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    cols = {r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'users'").fetchall()}
    assert tables == {"users", "segments"} and cols == {"user_id", "region", "device"}


def test_runtime_error_returns_error_status_not_exception(data_dir):
    p = pipe([c("SELECT CAST('abc' AS INTEGER) AS x FROM segments")], data_dir=data_dir)
    r = p.ask("segments as integers")
    assert r.status == "error" and r.reasons


# --- semantic judge, retrieval, parsing ---------------------------------------------------
def test_semantic_judge_can_reject_valid_but_wrong_sql():
    sem = ScriptedLLM([Completion("NO\nIt counts all users, not high-value ones.", SURE),
                       Completion("YES\nCorrect filter.", SURE)])
    p = pipe([c("SELECT COUNT(*) FROM segments"),
              c("SELECT COUNT(*) FROM segments WHERE segment = 'high_value'")], semantic=sem)
    r = p.ask("how many high value users")
    assert r.status == "ok" and r.attempts == 2 and "not high-value" in p.builder.prompts[1]


def test_schema_retrieval_picks_relevant_tables():
    names = [t.name for t in retrieve("video views by content category last week", k=2)]
    assert names[0] == "events"
    assert "segments" in [t.name for t in retrieve("which segment has the most users", k=2)]


def test_extract_sql_strips_fences_and_semicolons():
    assert extract_sql("Here you go:\n```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("SELECT 2;") == "SELECT 2"


def test_every_gold_query_in_eval_set_passes_judge_and_runs(data_dir):
    import json

    from nl2sql.rbac import ADMIN
    questions = json.loads((ROOT / "eval" / "questions.json").read_text())
    data, schema = with_data(data_dir), schema_only()
    for q in questions:
        v = judge.check(q["gold"], q["question"], ADMIN, schema)
        assert v.ok, (q["id"], v.reasons)
        execute(data, q["gold"])
