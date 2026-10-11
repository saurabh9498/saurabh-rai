"""
Schema catalog and schema retrieval (the "schema RAG" step).

An analytics warehouse has far more tables than fit usefully in a prompt, and every
irrelevant table is a chance for the model to join the wrong thing. So the Builder only
sees the few tables most relevant to the question. Retrieval here is a dependency-free
bag-of-words cosine over table and column descriptions; with hundreds of tables the same
interface would sit on an embedding index.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    description: str


@dataclass(frozen=True)
class Table:
    name: str
    description: str
    columns: tuple[Column, ...]

    def ddl(self) -> str:
        cols = ",\n  ".join(f"{c.name} {c.type}" for c in self.columns)
        return f"CREATE TABLE {self.name} (\n  {cols}\n);"

    def prompt_block(self, allowed: set[str] | None = None) -> str:
        cols = [c for c in self.columns if allowed is None or c.name in allowed]
        lines = [f"-- {self.description}", f"TABLE {self.name}("]
        lines += [f"  {c.name} {c.type},  -- {c.description}" for c in cols]
        return "\n".join(lines) + "\n)"


CATALOG: tuple[Table, ...] = (
    Table("users", "One row per user: profile attributes.", (
        Column("user_id", "BIGINT", "user identifier"),
        Column("region", "VARCHAR", "US region: northeast, southeast, midwest, southwest, west"),
        Column("device", "VARCHAR", "primary device: ctv, mobile, desktop, tablet"),
        Column("signup_days_ago", "INTEGER", "days since the user signed up"),
    )),
    Table("events", "One row per ad or content event: impressions, clicks, video views, conversions.", (
        Column("user_id", "BIGINT", "user identifier"),
        Column("event_type", "VARCHAR", "impression, click, video_view or conversion"),
        Column("category", "VARCHAR", "content category: sports, news, auto, travel, finance, entertainment, food"),
        Column("days_ago", "INTEGER", "how many days ago the event happened (0-89)"),
        Column("spend_usd", "DOUBLE", "purchase amount in dollars for conversions, else 0"),
    )),
    Table("segments", "Audience segment per user from the nightly segmentation job.", (
        Column("user_id", "BIGINT", "user identifier"),
        Column("segment", "VARCHAR", "high_value, engaged_viewer, lapsed or casual"),
        Column("spend_90d", "DOUBLE", "total conversion spend in the last 90 days"),
        Column("views_30d", "BIGINT", "video views in the last 30 days"),
    )),
    Table("user_pii", "Restricted: personal contact details for users.", (
        Column("user_id", "BIGINT", "user identifier"),
        Column("email", "VARCHAR", "email address"),
        Column("postal_code", "VARCHAR", "postal code"),
    )),
)

BY_NAME = {t.name: t for t in CATALOG}
_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> Counter:
    words = _WORD.findall(text.lower().replace("_", " "))
    return Counter(w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words)  # crude stemming


def _cosine(a: Counter, b: Counter) -> float:
    dot = sum(a[k] * b[k] for k in a.keys() & b.keys())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def restrict(table: Table, allowed: set[str] | frozenset[str]) -> Table:
    """The table as a given role sees it: only permitted columns."""
    return Table(table.name, table.description, tuple(c for c in table.columns if c.name in allowed))


def retrieve(question: str, tables: tuple[Table, ...] = CATALOG, k: int = 2) -> list[Table]:
    """Top-k tables by similarity between the question and each table's text."""
    q = _tokens(question)
    scored = []
    for t in tables:
        text = " ".join([t.name, t.description] + [f"{c.name} {c.description}" for c in t.columns])
        scored.append((_cosine(q, _tokens(text)), t.name, t))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [t for score, _, t in scored[:k] if score > 0]
