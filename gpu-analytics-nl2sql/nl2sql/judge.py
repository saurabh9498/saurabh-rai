"""
The Judge: a validation gate between generated SQL and the database.

Nothing the Builder writes reaches the warehouse until it passes, in order:

  1. parses as exactly one statement
  2. is read-only: a SELECT (or WITH ... SELECT); no DML, DDL, COPY, ATTACH, PRAGMA, SET,
     and no table functions or file paths (read_csv, read_parquet, 'file.csv')
  3. touches only tables and columns the user's role may read (nl2sql/rbac.py)
  4. plans against the real schema - an EXPLAIN dry-run, no data scanned
  5. optional semantic check: a second model call asks whether the SQL answers the question

Checks 1-4 are deterministic and cheap, so they run first; the model-based check runs only
on SQL that is already valid. A rejection carries its reasons back to the Builder for one
retry, which is how most "almost right" queries get fixed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from .llm import LLM
from .rbac import Role, violations
from .warehouse import dry_run

FORBIDDEN = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter, exp.Copy,
             exp.Command, exp.Set, exp.Pragma, exp.Merge, exp.TruncateTable)

SEMANTIC_PROMPT = """You are reviewing SQL written for an analyst.
Question: {question}
SQL:
{sql}
Does this SQL correctly answer the question? Answer YES or NO on the first line,
then one sentence explaining why."""


@dataclass
class Verdict:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    stage: str = "passed"


def check(sql: str, question: str, role: Role, schema_con: duckdb.DuckDBPyConnection,
          semantic_llm: LLM | None = None) -> Verdict:
    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except ParseError as e:
        return Verdict(False, [f"does not parse: {str(e).splitlines()[0]}"], "parse")
    if len(statements) != 1:
        return Verdict(False, [f"expected one statement, got {len(statements)}"], "parse")
    tree = statements[0]

    if not isinstance(tree, exp.Query) or any(tree.find_all(*FORBIDDEN)):
        return Verdict(False, ["only read-only SELECT queries are allowed"], "read_only")
    for t in tree.find_all(exp.Table):
        if not isinstance(t.this, exp.Identifier):
            return Verdict(False, ["table functions and file reads are not allowed"], "read_only")

    denied = violations(sql, role)
    if denied:
        return Verdict(False, [f"role '{role.name}' may not read {d}" for d in denied], "rbac")

    err = dry_run(schema_con, sql)
    if err:
        return Verdict(False, [f"does not plan: {err}"], "dry_run")

    if semantic_llm is not None:
        reply = semantic_llm.complete(SEMANTIC_PROMPT.format(question=question, sql=sql)).text.strip()
        first, _, why = reply.partition("\n")
        if not first.strip().upper().startswith("YES"):
            return Verdict(False, [f"semantic check: {why.strip() or first.strip()}"], "semantic")
    return Verdict(True)
