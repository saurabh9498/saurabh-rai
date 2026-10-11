"""
Walk-through of the pipeline against the generated warehouse, with a scripted model.

Shows the four outcomes: a query that runs, one the Judge fixes on retry, one that is
too uncertain to run, and one blocked by role-based access control.

Run after data/generate_events.py and segmentation/segment.py:
    python demo.py
"""
from __future__ import annotations

import pathlib

from nl2sql.llm import Completion, ScriptedLLM
from nl2sql.pipeline import Pipeline
from nl2sql.rbac import ANALYST, MARKETING

DATA = pathlib.Path(__file__).resolve().parent / "data" / "generated"
SURE, VAGUE = [-0.01] * 20, [-0.01] * 15 + [-2.5] * 5


def show(title: str, result) -> None:
    print(f"\n== {title}\nstatus: {result.status} (attempts: {result.attempts})")
    if result.sql:
        print(f"sql:    {result.sql}")
    if result.reasons:
        print(f"why:    {'; '.join(result.reasons)}")
    if result.clarifying_question:
        print(f"ask:    {result.clarifying_question}")
    for row in result.rows[:6]:
        print(f"        {row}")


def main() -> None:
    def run(role, script, question):
        return Pipeline.for_role(ScriptedLLM(script), role, DATA).ask(question)

    show("runs", run(ANALYST, [Completion(
        "SELECT segment, COUNT(*) AS users FROM segments GROUP BY segment ORDER BY users DESC", SURE)],
        "how many users are in each segment"))
    show("fixed on retry", run(ANALYST, [
        Completion("SELECT category, COUNT(*) FROM events WHERE type = 'video_view' GROUP BY category", SURE),
        Completion("SELECT category, COUNT(*) AS views FROM events WHERE event_type = 'video_view' "
                   "GROUP BY category ORDER BY views DESC", SURE)],
        "video views by content category"))
    show("too uncertain to run", run(ANALYST, [Completion(
        "SELECT user_id FROM events ORDER BY spend_usd DESC LIMIT 10", VAGUE)],
        "top customers last month"))
    show("blocked by role", run(MARKETING, [Completion(
        "SELECT u.signup_days_ago FROM users u", SURE)],
        "how long ago did lapsed users sign up"))


if __name__ == "__main__":
    main()
