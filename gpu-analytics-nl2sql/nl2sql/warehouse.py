"""
DuckDB warehouse connections, one set per role.

Each role gets connections that contain ONLY the tables and columns it may read. This is
the security boundary for access control: a query cannot leak a column that does not
exist in its connection, however it is written (see nl2sql/rbac.py for why the parser
check alone is not enough).

  * schema_only(role)   empty tables with the role's columns - for the Judge's EXPLAIN
                        dry-run, which catches unknown or forbidden columns without
                        scanning any data
  * with_data(dir, role) the same tables loaded from Parquet - for execution

Both are locked after setup: external file access is disabled and the configuration is
frozen, so even a query like `FROM read_csv('/etc/passwd')` that slipped past the Judge
cannot touch the filesystem, and no query can re-enable it.
"""
from __future__ import annotations

import pathlib
import threading

import duckdb

from .rbac import ADMIN, Role
from .schema import CATALOG


def _lock(con: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    con.execute("SET enable_external_access = false")
    con.execute("SET lock_configuration = true")
    return con


def _role_columns(role: Role):
    for t in CATALOG:
        if role.allows(t.name):
            yield t, [c for c in t.columns if role.allows(t.name, c.name)]


def schema_only(role: Role = ADMIN) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    for t, cols in _role_columns(role):
        con.execute(f"CREATE TABLE {t.name} ({', '.join(f'{c.name} {c.type}' for c in cols)})")
    return _lock(con)


def with_data(data_dir: pathlib.Path, role: Role = ADMIN) -> duckdb.DuckDBPyConnection:
    """Role-scoped tables loaded from Parquet where files exist; empty tables otherwise."""
    con = duckdb.connect(":memory:")
    for t, cols in _role_columns(role):
        f = data_dir / f"{t.name}.parquet"
        if f.exists():
            select = ", ".join(f"CAST({c.name} AS {c.type}) AS {c.name}" for c in cols)
            con.execute(f"CREATE TABLE {t.name} AS SELECT {select} FROM read_parquet('{f.as_posix()}')")
        else:
            con.execute(f"CREATE TABLE {t.name} ({', '.join(f'{c.name} {c.type}' for c in cols)})")
    return _lock(con)


def dry_run(con: duckdb.DuckDBPyConnection, sql: str) -> str | None:
    """Return an error message if the SQL would not plan, else None."""
    try:
        con.execute(f"EXPLAIN {sql}")
        return None
    except duckdb.Error as e:
        return str(e).splitlines()[0]


def execute(con: duckdb.DuckDBPyConnection, sql: str, row_limit: int = 1000, timeout_s: float = 10.0):
    """Run with a row cap and a wall-clock timeout (DuckDB has no statement timeout).

    The row cap limits what is returned, not the work done; the timeout bounds the work.
    """
    timer = threading.Timer(timeout_s, con.interrupt)
    timer.start()
    try:
        rel = con.execute(f"SELECT * FROM ({sql}) AS q LIMIT {int(row_limit)}")
        cols = [d[0] for d in rel.description]
        return cols, rel.fetchall()
    finally:
        timer.cancel()
