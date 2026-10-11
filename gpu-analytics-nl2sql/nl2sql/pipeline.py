"""
Question in, rows (or a clarifying question) out.

    retrieve schema -> filter by role -> Builder writes SQL -> confidence check
        low confidence      -> ask the user a clarifying question; run nothing
        confident           -> Judge -> rejected? feed reasons back, retry once
                                     -> passed   -> execute with row cap and timeout

The connections passed in must be scoped to the same role (Pipeline.for_role builds them),
because the database, not the parser, is the access-control boundary.
"""
from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field

import duckdb

from . import judge
from .confidence import Confidence, score
from .llm import LLM
from .rbac import Role
from .schema import CATALOG, restrict, retrieve
from .warehouse import execute, schema_only, with_data

BUILDER_PROMPT = """Write one DuckDB SQL query that answers the question.
Use only these tables and columns:

{schema}

Question: {question}
{feedback}Return only the SQL."""

FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)


def extract_sql(text: str) -> str:
    m = FENCE.search(text)
    return (m.group(1) if m else text).strip().rstrip(";").strip()


@dataclass
class Result:
    status: str                              # "ok" | "clarify" | "rejected" | "error"
    sql: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    clarifying_question: str | None = None
    reasons: list[str] = field(default_factory=list)
    attempts: int = 0
    confidence: Confidence | None = None


@dataclass
class Pipeline:
    builder: LLM
    role: Role
    schema_con: duckdb.DuckDBPyConnection
    data_con: duckdb.DuckDBPyConnection | None = None
    semantic_judge: LLM | None = None
    mean_threshold: float = 0.80
    min_threshold: float = 0.20
    max_attempts: int = 2
    tables_k: int = 2

    @classmethod
    def for_role(cls, builder: LLM, role: Role, data_dir: pathlib.Path | None = None, **kw) -> "Pipeline":
        data = with_data(data_dir, role) if data_dir is not None else None
        return cls(builder=builder, role=role, schema_con=schema_only(role), data_con=data, **kw)

    def schema_for(self, question: str) -> str:
        visible = tuple(restrict(t, self.role.tables[t.name]) for t in CATALOG if self.role.allows(t.name))
        return "\n\n".join(t.prompt_block() for t in retrieve(question, visible, k=self.tables_k))

    def ask(self, question: str) -> Result:
        schema = self.schema_for(question)
        if not schema:
            return Result("rejected", reasons=["no tables this role may read match the question"])
        feedback, reasons = "", []
        for attempt in range(1, self.max_attempts + 1):
            comp = self.builder.complete(BUILDER_PROMPT.format(schema=schema, question=question, feedback=feedback))
            sql, conf = extract_sql(comp.text), score(comp.token_logprobs)
            if conf.low(self.mean_threshold, self.min_threshold):
                return Result("clarify", sql=sql, attempts=attempt, confidence=conf,
                              clarifying_question=clarify(question))
            verdict = judge.check(sql, question, self.role, self.schema_con, self.semantic_judge)
            if verdict.ok:
                if self.data_con is None:
                    return Result("ok", sql=sql, attempts=attempt, confidence=conf)
                try:
                    cols, rows = execute(self.data_con, sql)
                except duckdb.Error as e:   # runtime error or timeout: report it, don't crash
                    return Result("error", sql=sql, reasons=[str(e).splitlines()[0]],
                                  attempts=attempt, confidence=conf)
                return Result("ok", sql=sql, columns=cols, rows=rows, attempts=attempt, confidence=conf)
            reasons = verdict.reasons
            if verdict.stage in ("rbac", "read_only"):
                break                         # policy rejections are final; do not retry around them
            feedback = "A previous attempt was rejected: " + "; ".join(reasons) + "\nFix it.\n"
        return Result("rejected", sql=sql, reasons=reasons, attempts=attempt, confidence=conf)


def clarify(question: str) -> str:
    """Generic clarifying prompt; a production system asks about the specific weak token."""
    return (f'"{question}" could be read more than one way. Which metric, time window and '
            "segment do you mean?")
