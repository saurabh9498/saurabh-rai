"""
Role-based access control, enforced in three places.

1. Before generation: the Builder's prompt only contains tables and columns the role may
   read.
2. After generation: the SQL is parsed and checked against the role. This gives a clear
   rejection reason the Builder and the user can act on.
3. In the database - the actual security boundary. Each role queries its own DuckDB
   connection that only contains the tables and columns it may read
   (nl2sql/warehouse.py). Restricted data is not hidden from the query; it is absent.

Why layer 3 is necessary: SQL has many ways to reach a column without naming it - case
variants (SIGNUP_DAYS_AGO), whole-row references (SELECT u FROM users u), regex column
selection (COLUMNS('sign.*')), positional references (#4), and alias renaming
(users AS t(a, b, c, sd)). The parser check rejects all of those patterns, but a security
boundary cannot depend on enumerating syntax. Layer 3 does not.
"""
from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from .schema import BY_NAME


@dataclass(frozen=True)
class Role:
    name: str
    tables: dict[str, frozenset[str]]       # table -> allowed columns (lower case)

    def allows(self, table: str, column: str | None = None) -> bool:
        table = table.lower()
        if table not in self.tables:
            return False
        return column is None or column.lower() in self.tables[table]


def _all(table: str) -> frozenset[str]:
    return frozenset(c.name for c in BY_NAME[table].columns)


ANALYST = Role("analyst", {t: _all(t) for t in ("users", "events", "segments")})
MARKETING = Role("marketing", {"segments": _all("segments"), "users": frozenset({"user_id", "region", "device"})})
ADMIN = Role("admin", {t: _all(t) for t in BY_NAME})


def _in_count(star: exp.Star) -> bool:
    return isinstance(star.parent, exp.Count)


def violations(sql: str, role: Role) -> list[str]:
    """Reasons the SQL may not run for this role (empty list = no parser-level problem)."""
    tree = sqlglot.parse_one(sql, read="duckdb")
    problems: list[str] = []

    if any(tree.find_all(exp.Columns)):
        problems.append("COLUMNS(...) selection is not allowed")
    if any(tree.find_all(exp.PositionalColumn)):
        problems.append("positional column references are not allowed")

    alias_to_table: dict[str, str] = {}
    tables_used: set[str] = set()
    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    for t in tree.find_all(exp.Table):
        name = t.name.lower()
        if name in cte_names:
            continue
        tables_used.add(name)
        alias_to_table[t.alias_or_name.lower()] = name
        alias = t.args.get("alias")
        if isinstance(alias, exp.TableAlias) and alias.args.get("columns"):
            problems.append(f"renaming columns of {name} in a table alias is not allowed")
    problems += [f"table {t}" for t in sorted(tables_used) if not role.allows(t)]

    for col in tree.find_all(exp.Column):
        name = col.name.lower()
        if not name or isinstance(col.this, exp.Star):
            continue
        if not col.table and name in alias_to_table:
            problems.append(f"whole-row reference to {alias_to_table[name]}")
            continue
        if col.table:
            owners = [alias_to_table.get(col.table.lower(), col.table.lower())]
        else:
            owners = [t for t in tables_used if t in BY_NAME and name in _all(t)]
            if not owners:
                continue  # an output alias or CTE column; the dry-run catches unknown names
        if not any(role.allows(o, name) for o in owners):
            problems.append(f"column {owners[0]}.{name}")

    if any(not _in_count(s) for s in tree.find_all(exp.Star)):
        for t in tables_used:
            if role.allows(t) and role.tables[t] != _all(t):
                problems.append(f"SELECT * on partially restricted table {t}")
    return sorted(set(problems))
