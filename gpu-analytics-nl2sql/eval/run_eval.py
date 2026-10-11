"""
Does the Judge reduce failed queries? Builder-only vs Builder + Judge, on a real model.

Designed harness - NOT run in this repo (there is no model endpoint here), so this repo
reports no accuracy or failure-rate numbers. Point it at any OpenAI-compatible endpoint
that returns log-probabilities (vLLM, TensorRT-LLM, OpenAI):

    NL2SQL_BASE_URL=http://localhost:8000/v1 NL2SQL_MODEL=<model> python eval/run_eval.py

Every question ends in exactly one outcome per mode:
    correct    - result set equals the gold query's (order-insensitive unless the gold
                 query's outer SELECT has an ORDER BY)
    wrong      - ran but returned a different result
    error      - did not run (error, timeout, or rejected by the Judge)
    clarified  - asked a clarifying question instead (Builder + Judge only)

All four rates are reported for both modes, so a clarification is never silently counted
as a success. Calibrate the confidence thresholds on a separate split first.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nl2sql.llm import OpenAICompatibleLLM  # noqa: E402
from nl2sql.pipeline import BUILDER_PROMPT, Pipeline, extract_sql  # noqa: E402
from nl2sql.rbac import ADMIN  # noqa: E402
from nl2sql.warehouse import execute, with_data  # noqa: E402


def same_result(a: list[tuple], b: list[tuple], ordered: bool) -> bool:
    norm = lambda rows: [tuple(round(v, 6) if isinstance(v, float) else v for v in r) for r in rows]  # noqa: E731
    a, b = norm(a), norm(b)
    return a == b if ordered else sorted(map(repr, a)) == sorted(map(repr, b))


def main() -> None:
    questions = json.loads((ROOT / "eval" / "questions.json").read_text())
    data_dir = ROOT / "data" / "generated"
    data = with_data(data_dir)
    llm = OpenAICompatibleLLM()
    pipe = Pipeline.for_role(llm, ADMIN, data_dir)
    rows = []
    for q in questions:
        _, gold = execute(data, q["gold"])
        tail = q["gold"].lower().rsplit("order by", 1)
        ordered = len(tail) == 2 and ")" not in tail[1]      # ORDER BY on the outer query only

        # Builder only: one attempt, executed as-is, no gate.
        sql = extract_sql(llm.complete(BUILDER_PROMPT.format(
            schema=pipe.schema_for(q["question"]), question=q["question"], feedback="")).text)
        try:
            _, got = execute(data, sql)
            b = "correct" if same_result(got, gold, ordered) else "wrong"
        except Exception:
            b = "error"

        r = pipe.ask(q["question"])
        if r.status == "clarify":
            j = "clarified"
        elif r.status == "ok":
            j = "correct" if same_result(r.rows, gold, ordered) else "wrong"
        else:
            j = "error"
        rows.append({"id": q["id"], "builder_only": b, "builder_judge": j, "judge_attempts": r.attempts})

    n = len(rows)
    outcomes = ("correct", "wrong", "error", "clarified")
    summary = {"questions": n}
    for mode in ("builder_only", "builder_judge"):
        summary[mode] = {o: sum(r[mode] == o for r in rows) / n for o in outcomes}
    print(json.dumps({"summary": summary, "per_question": rows}, indent=2))


if __name__ == "__main__":
    main()
