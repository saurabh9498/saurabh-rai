"""
A stand-in for the systems the agent talks to: patient records and the scheduling
calendar. In production these are EHR scheduling APIs; here they are in memory and
deterministic so the agent can be tested end to end.
"""
from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass, field

DEPARTMENTS = {
    "primary care": ("primary care", "family medicine", "general practitioner", "gp", "check-up", "checkup", "physical"),
    "cardiology": ("cardiology", "cardiologist", "heart doctor", "heart"),
    "dermatology": ("dermatology", "dermatologist", "skin"),
    "pediatrics": ("pediatrics", "pediatrician", "kids doctor", "child"),
    "physical therapy": ("physical therapy", "physiotherapy", "pt appointment", "physical therapist"),
}


@dataclass(frozen=True)
class Patient:
    patient_id: str
    first: str
    last: str
    dob: dt.date


@dataclass(frozen=True)
class Slot:
    department: str
    start: dt.datetime

    def label(self) -> str:
        return self.start.strftime("%A, %B %-d at %-I:%M %p")


@dataclass
class Clinic:
    today: dt.date
    patients: dict[str, Patient] = field(default_factory=dict)
    free: set[Slot] = field(default_factory=set)
    booked: dict[Slot, str] = field(default_factory=dict)       # slot -> patient_id

    @classmethod
    def demo(cls, today: dt.date = dt.date(2026, 3, 16), seed: int = 7) -> "Clinic":
        rng = random.Random(seed)
        c = cls(today=today)
        for p in (Patient("P001", "Maria", "Lopez", dt.date(1985, 3, 14)),
                  Patient("P002", "James", "Okafor", dt.date(1972, 11, 2)),
                  Patient("P003", "Priya", "Raman", dt.date(1990, 6, 30))):
            c.patients[p.patient_id] = p
        for d in range(1, 15):
            day = today + dt.timedelta(days=d)
            if day.weekday() >= 5:
                continue
            for dept in DEPARTMENTS:
                for h in range(9, 17):
                    for m in (0, 30):
                        if rng.random() < 0.35:            # ~35% of the grid is open
                            c.free.add(Slot(dept, dt.datetime.combine(day, dt.time(h, m))))
        # An existing appointment so cancellation can be demonstrated.
        existing = Slot("dermatology", dt.datetime.combine(today + dt.timedelta(days=3), dt.time(14, 0)))
        c.free.discard(existing)
        c.booked[existing] = "P001"
        return c

    def verify(self, full_name: str, dob: dt.date) -> Patient | None:
        parts = full_name.lower().split()
        if len(parts) < 2:
            return None
        for p in self.patients.values():
            if p.first.lower() == parts[0] and p.last.lower() == parts[-1] and p.dob == dob:
                return p
        return None

    def search(self, department: str, day: dt.date | None, part: str | None, limit: int = 3) -> list[Slot]:
        def ok(s: Slot) -> bool:
            if s.department != department or s.start.date() <= self.today:
                return False
            if day and s.start.date() != day:
                return False
            h = s.start.hour
            return {"morning": h < 12, "afternoon": 12 <= h < 17, None: True}.get(part, True)
        return sorted((s for s in self.free if ok(s)), key=lambda s: s.start)[:limit]

    def book(self, slot: Slot, patient_id: str) -> bool:
        if slot not in self.free:           # taken since it was offered
            return False
        self.free.discard(slot)
        self.booked[slot] = patient_id
        return True

    def upcoming(self, patient_id: str) -> list[Slot]:
        return sorted((s for s, p in self.booked.items() if p == patient_id and s.start.date() > self.today),
                      key=lambda s: s.start)

    def cancel(self, slot: Slot, patient_id: str) -> bool:
        if self.booked.get(slot) != patient_id:
            return False
        del self.booked[slot]
        self.free.add(slot)
        return True
