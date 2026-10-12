"""
Offline-first store-and-forward for an unattended in-store device.

The analytics must keep working when the store's uplink is down - for minutes, or for a
long weekend. Events are written to a local SQLite file first (it survives power loss and
reboots) and forwarded when the link is back.

Guarantees, each tested:
  * Nothing is lost while there is space. Delivery is at-least-once: an event is deleted
    only after the server acknowledges it.
  * Duplicates are harmless. Every event has a stable ID and the device ignores a repeated
    write. The server is expected to de-duplicate on the same ID (not implemented here), so
    a retry after a lost acknowledgement does not double-count.
  * When the disk budget runs out, the least important, oldest events go first: raw
    per-frame metadata before per-minute counts, and alerts last.
  * Forwarding preserves time order and stops at the first failure, so a flaky link never
    reorders or skips events.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Callable, Iterable

# Lower number = dropped first when the buffer is full.
PRIORITY = {"frame_meta": 0, "count": 1, "alert": 2}


@dataclass(frozen=True)
class Event:
    event_id: str
    ts: float
    kind: str              # "frame_meta" | "count" | "alert"
    payload: bytes


class StoreAndForward:
    def __init__(self, path: str = ":memory:", capacity_bytes: int = 64 * 2**20):
        self.capacity = capacity_bytes
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL") if path != ":memory:" else None
        self.db.execute("""CREATE TABLE IF NOT EXISTS events(
            id TEXT PRIMARY KEY, ts REAL, kind TEXT, prio INTEGER, size INTEGER, payload BLOB)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS by_eviction ON events(prio, ts)")
        self.db.execute("CREATE INDEX IF NOT EXISTS by_time ON events(ts, id)")
        self.dropped: dict[str, int] = {k: 0 for k in PRIORITY}

    def used_bytes(self) -> int:
        return self.db.execute("SELECT COALESCE(SUM(size), 0) FROM events").fetchone()[0]

    def __len__(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    def put(self, e: Event) -> bool:
        """Store an event. Returns False if it was a duplicate or could not fit."""
        size = len(e.payload) + 64                       # payload + row overhead estimate
        prio = PRIORITY[e.kind]
        with self.db:
            if self.db.execute("SELECT 1 FROM events WHERE id = ?", (e.event_id,)).fetchone():
                return False
            need = self.used_bytes() + size - self.capacity
            if need > 0:
                # Decide BEFORE deleting anything: only events of equal or lower priority may be
                # evicted, and if they cannot free enough room the new event is refused intact.
                evictable = self.db.execute("SELECT COALESCE(SUM(size), 0) FROM events WHERE prio <= ?",
                                            (prio,)).fetchone()[0]
                if evictable < need:
                    self.dropped[e.kind] += 1
                    return False
                for vid, vkind, vsize in self.db.execute(
                        "SELECT id, kind, size FROM events WHERE prio <= ? ORDER BY prio ASC, ts ASC",
                        (prio,)).fetchall():
                    if need <= 0:
                        break
                    self.db.execute("DELETE FROM events WHERE id = ?", (vid,))
                    self.dropped[vkind] += 1
                    need -= vsize
            self.db.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
                            (e.event_id, e.ts, e.kind, prio, size, e.payload))
        return True

    def drain(self, send: Callable[[list[Event]], bool], batch: int = 500) -> int:
        """Forward in time order. `send` returns True when the server acknowledged the batch."""
        sent = 0
        while True:
            rows = self.db.execute("SELECT id, ts, kind, payload FROM events ORDER BY ts ASC, id ASC LIMIT ?",
                                   (batch,)).fetchall()
            if not rows:
                return sent
            events = [Event(r[0], r[1], r[2], r[3]) for r in rows]
            if not send(events):
                return sent                               # link down: keep everything, retry later
            with self.db:
                self.db.executemany("DELETE FROM events WHERE id = ?", [(e.event_id,) for e in events])
            sent += len(events)


def bytes_per_hour(rates: Iterable[tuple[str, float, int]]) -> float:
    """rates: (kind, events per hour, payload bytes). Includes the per-row overhead."""
    return sum(n * (size + 64) for _, n, size in rates)


def hours_of_outage(capacity_bytes: int, rates: Iterable[tuple[str, float, int]]) -> float:
    return capacity_bytes / bytes_per_hour(list(rates))
