"""
Latched circuit breaker: GPU health fault -> withdraw the hub's anycast prefix.

Every edge hub announces the same anycast prefix over BGP, so the network sends each bid
to the nearest hub. If a hub's GPU fails but the hub keeps announcing, bids keep arriving
and time out. This breaker withdraws the announcement on a fatal fault, and the network
reroutes those bids to the next-nearest healthy hub.

The dangerous failure mode of any automation that touches routing is flapping: a GPU
oscillating between healthy and unhealthy would make the prefix appear and disappear,
destabilising routing for every hub. Three guards prevent that:

  1. Persistence  - a fault must hold for `persist_s` (15 s) before it counts. In production
                    Alertmanager's `for: 15s` does this; the breaker re-checks it so it is
                    safe on its own.
  2. Latching     - once tripped, the breaker stays OPEN. A fault that clears does NOT
                    re-announce the prefix. Hardware that failed once under load usually
                    fails again.
  3. Slow restore - re-announcing needs either `cooldown_s` (30 min) of continuous health,
                    or an explicit operator reset. Any new fault restarts the cooldown.

State survives restarts: with `state_path` set, every transition is written to disk and
reloaded at startup. Without it, a restart would forget that the prefix is withdrawn and
nothing would ever re-announce it.

The breaker is pure logic with an injected clock and switch client, so it is fully
testable without a switch (see tests/test_rtb_edge_scale_out.py).
"""
from __future__ import annotations

import enum
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

log = logging.getLogger("breaker")


class SwitchClient(Protocol):
    def withdraw(self, prefix: str) -> None: ...
    def announce(self, prefix: str) -> None: ...


class State(enum.Enum):
    CLOSED = "closed"            # healthy, prefix announced
    OPEN = "open"                # tripped, prefix withdrawn (latched)


# Fault classes that trip the breaker.
#   Xid 79 = GPU fell off the bus, 48 = double-bit ECC, 95 = uncontained ECC. Any of these
#   means the replica is gone or about to be. (Xid 94, contained ECC, usually recovers and
#   pages a human instead; Xid 74 is NVLink, which these PCIe cards do not have.)
FATAL_XIDS = frozenset({48, 79, 95})
#   Sustained thermal throttling is treated as fatal for TP>1 groups: over PCIe, three healthy
#   GPUs wait on the slowest one at every step, so one hot card stalls the whole group.
THERMAL = "thermal_throttle"
#   A GPU that stops reporting (fell off the bus) or a dead exporter: no evidence of health.
MISSING = "gpu_missing"


@dataclass
class Fault:
    gpu: str
    kind: str                    # "xid" or THERMAL
    xid: int | None = None

    def is_fatal(self) -> bool:
        return self.kind in (THERMAL, MISSING) or (self.kind == "xid" and self.xid in FATAL_XIDS)


@dataclass
class LatchedBreaker:
    prefix: str
    switch: SwitchClient
    persist_s: float = 15.0
    cooldown_s: float = 30 * 60.0
    clock: Callable[[], float] = time.time          # wall clock, so persisted times stay valid
    state_path: str | None = None
    state: State = State.CLOSED
    _first_seen: dict[str, float] = field(default_factory=dict)
    _active: set[str] = field(default_factory=set)
    _healthy_since: float | None = None
    events: list[tuple[float, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.state_path and os.path.exists(self.state_path):
            with open(self.state_path) as f:
                saved = json.load(f)
            self.state = State(saved["state"])
            self._healthy_since = saved.get("healthy_since")
            self._log(f"loaded persisted state {self.state.value}")

    # --- inputs -----------------------------------------------------------------------
    def fault_firing(self, fault: Fault, already_persisted: bool = False) -> None:
        """A fault is currently present.

        `already_persisted=True` means the source has already debounced it - an
        Alertmanager alert only fires after its rule's `for: 15s`. Raw signals (for
        example a direct DCGM poll) leave it False and the breaker applies `persist_s`.
        """
        if not fault.is_fatal():
            return
        now = self.clock()
        key = f"{fault.gpu}:{fault.kind}:{fault.xid}"
        self._active.add(key)
        self._first_seen.setdefault(key, now - (self.persist_s if already_persisted else 0.0))
        self._healthy_since = None                       # any live fault restarts cooldown
        if self.state is State.CLOSED and now - self._first_seen[key] >= self.persist_s:
            self._trip(f"{key} persisted >= {self.persist_s:.0f}s")

    def fault_resolved(self, fault: Fault) -> None:
        key = f"{fault.gpu}:{fault.kind}:{fault.xid}"
        self._active.discard(key)
        self._first_seen.pop(key, None)
        if not self._active and self.state is State.OPEN and self._healthy_since is None:
            self._healthy_since = self.clock()
            self._save()
            self._log("all faults cleared; cooldown started (prefix stays withdrawn)")

    def tick(self) -> None:
        """Periodic check: restore only after a full cooldown of continuous health."""
        if (self.state is State.OPEN and not self._active and self._healthy_since is not None
                and self.clock() - self._healthy_since >= self.cooldown_s):
            self._restore("cooldown elapsed with no faults")

    def operator_reset(self, who: str) -> None:
        """Manual override after a human has inspected the node."""
        if self._active:
            raise RuntimeError("refusing reset: faults still active")
        if self.state is State.OPEN:
            self._restore(f"operator reset by {who}")

    # --- transitions ------------------------------------------------------------------
    def _trip(self, why: str) -> None:
        self.switch.withdraw(self.prefix)
        self.state = State.OPEN
        self._save()
        self._log(f"TRIP: withdrew {self.prefix} ({why})")

    def _restore(self, why: str) -> None:
        self.switch.announce(self.prefix)
        self.state = State.CLOSED
        self._healthy_since = None
        self._save()
        self._log(f"RESTORE: announced {self.prefix} ({why})")

    def _save(self) -> None:
        if not self.state_path:
            return
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"state": self.state.value, "healthy_since": self._healthy_since}, f)
        os.replace(tmp, self.state_path)                 # atomic on POSIX

    def _log(self, msg: str) -> None:
        self.events.append((self.clock(), msg))
        log.warning(msg)
