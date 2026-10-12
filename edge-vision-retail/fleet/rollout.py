"""
Staged updates for hundreds of devices nobody can walk up to.

A bad model or runtime release on unattended hardware is expensive: no one is on site to
reinstall it. So a release goes out in waves, and each wave must pass a health gate before
the next one starts:

    canary 1%  ->  10%  ->  50%  ->  100%

Each device keeps two slots (A/B): the new version installs into the inactive slot, and a
rollback is a local switch back to the old slot - it needs no download, so it works even if
the store's link is down at that moment.

Health gate, per wave, compared with the same devices before the update:
  * the device came back healthy (no crash loop)
  * throughput (frames per second) did not drop more than 10%
  * the detection rate did not drift more than 15% - a cheap accuracy proxy that needs no labels

A failed gate rolls back every device that took the release and stops the rollout. Devices
that were offline during the rollout get the release when they reconnect only if it was
completed; a rolled-back release is never delivered late.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable

WAVES = (0.01, 0.10, 0.50, 1.00)


@dataclass
class Device:
    device_id: str
    online: bool = True
    active: str = "v1"
    standby: str | None = None


@dataclass(frozen=True)
class Health:
    healthy: bool
    fps: float
    detections_per_min: float


@dataclass
class Gate:
    max_fps_drop: float = 0.10
    max_detection_drift: float = 0.15

    def check(self, before: Health, after: Health) -> str | None:
        if not after.healthy:
            return "device unhealthy after update"
        if after.fps < before.fps * (1 - self.max_fps_drop):
            return f"fps dropped {1 - after.fps / before.fps:.0%}"
        drift = abs(after.detections_per_min - before.detections_per_min) / max(before.detections_per_min, 1e-9)
        if drift > self.max_detection_drift:
            return f"detection rate drifted {drift:.0%}"
        return None


def wave_of(device_id: str) -> float:
    """Stable position in [0, 1): the same device is always in the same wave."""
    return int(hashlib.sha256(device_id.encode()).hexdigest()[:8], 16) / 2**32


@dataclass
class Rollout:
    devices: list[Device]
    probe: Callable[[Device, str], Health]           # health of a device running a version
    gate: Gate = field(default_factory=Gate)
    log: list[str] = field(default_factory=list)
    status: str = "pending"                          # -> "completed" | "rolled_back"
    release: str | None = None

    def run(self, release: str) -> str:
        self.release = release
        upgraded: list[Device] = []
        done = 0.0
        for frac in WAVES:
            wave = [d for d in self.devices if done <= wave_of(d.device_id) < frac and d.online]
            for d in wave:
                before = self.probe(d, d.active)
                d.standby, d.active = d.active, release        # install into the B slot, switch
                upgraded.append(d)
                reason = self.gate.check(before, self.probe(d, release))
                if reason:
                    self.log.append(f"gate failed on {d.device_id} in the {frac:.0%} wave: {reason}")
                    self._rollback(upgraded)
                    self.status = "rolled_back"
                    return self.status
            self.log.append(f"wave {frac:.0%}: {len(wave)} devices passed")
            done = frac
        self.status = "completed"
        return self.status

    def _rollback(self, upgraded: list[Device]) -> None:
        for d in upgraded:
            d.active, d.standby = d.standby, d.active          # local slot switch, no download
        self.log.append(f"rolled back {len(upgraded)} devices")

    def on_reconnect(self, d: Device) -> None:
        """A device that was offline during the rollout catches up - only to a completed release."""
        d.online = True
        if self.status == "completed" and d.active != self.release:
            d.standby, d.active = d.active, self.release
