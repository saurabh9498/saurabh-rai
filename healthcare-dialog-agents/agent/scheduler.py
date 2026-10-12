"""
The scheduling agent: verify the patient, collect what they need, offer real slots,
confirm, then act - with safety and human handoff checked on every turn.

Data flow on each patient message:

    raw text --redact(prompt)--> understanding (rule-based or model)   no identity seen
        |
        +--> identity values looked up in the Vault                    only for verification
        |
        +--redact(log)--> audit trail                                   no identifiers or dates

Guardrails:
  * An urgent symptom ends the automated flow on any turn, before anything else runs.
  * Nothing is booked or cancelled without an explicit "yes" to a read-back.
  * A slot is re-checked at booking time; if it was taken meanwhile, new options are offered.
  * Three failed verifications, or two turns in a row without progress, hand off to staff.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Callable

from phi.redact import Vault, redact

from .audit import AuditLog, pseudonym
from .clinic import Clinic, Patient, Slot
from .nlu import Frame, is_urgent, understand

EMERGENCY = ("If you are having a medical emergency, please call 911 now. If you are thinking about "
             "harming yourself, you can call or text 988 at any time. I'm connecting you with a "
             "member of our staff.")
STAFF = "I'm connecting you with a member of our staff, who can help from here."


@dataclass
class Conversation:
    conv_id: str
    stage: str = "verify"            # verify -> collect -> offer -> confirm -> done | handoff
    vault: Vault = field(default_factory=Vault)
    patient: Patient | None = None
    intent: str | None = None
    department: str | None = None
    day: dt.date | None = None
    part: str | None = None
    options: list[Slot] = field(default_factory=list)
    pending: Slot | None = None
    verify_failures: int = 0
    verify_prompts: int = 0
    stalled_turns: int = 0
    handoff_reason: str | None = None
    turn: int = 0


@dataclass
class SchedulingAgent:
    clinic: Clinic
    audit: AuditLog = field(default_factory=AuditLog)
    nlu: Callable[[str, dt.date], Frame] = understand
    model_inputs: list[str] = field(default_factory=list)   # everything the language layer saw

    def handle(self, conv: Conversation, text: str) -> str:
        conv.turn += 1
        names = (f"{conv.patient.first} {conv.patient.last}", conv.patient.first, conv.patient.last) if conv.patient else ()
        verifying = conv.stage == "verify"
        prompt_text = redact(text, conv.vault, "prompt", names, verify=verifying)
        self.audit.append(conv.conv_id, "patient", "message",
                          redact(text, conv.vault, "log", names, verify=verifying), conv.turn)
        if is_urgent(text):                      # raw text, local check - before anything else,
            reply = self._handoff(conv, "urgent symptoms", EMERGENCY)    # even after "done"
            self.audit.append(conv.conv_id, "agent", conv.stage, reply, conv.turn)
            return reply
        self.model_inputs.append(prompt_text)
        f = self.nlu(prompt_text, self.clinic.today)

        if conv.stage in ("done", "handoff"):
            reply = "This conversation has ended. Please start a new one if you need anything else."
        elif f.human:
            reply = self._handoff(conv, "patient asked for a person", STAFF)
        else:
            before = (conv.stage, conv.intent, conv.department, conv.day, conv.part, conv.pending)
            reply = getattr(self, f"_{conv.stage}")(conv, f, prompt_text)
            after = (conv.stage, conv.intent, conv.department, conv.day, conv.part, conv.pending)
            # Verification has its own failure counter; stalls count from the request onward.
            progressed = after != before or before[0] == "verify"
            conv.stalled_turns = 0 if progressed else conv.stalled_turns + 1
            if conv.stalled_turns >= 2 and conv.stage not in ("done", "handoff"):
                reply = self._handoff(conv, "not understood twice", "Sorry, I'm not getting this right. " + STAFF)

        self.audit.append(conv.conv_id, "agent", conv.stage, redact(reply, conv.vault, "log", names), conv.turn)
        return conv.vault.restore(reply)

    # --- stages ------------------------------------------------------------------------
    def _verify(self, conv: Conversation, f: Frame, prompt_text: str) -> str:
        name_tok = re.search(r"\[NAME_\d+\]", prompt_text)
        dob_tok = re.search(r"\[DOB_\d+\]", prompt_text)
        if not (name_tok and dob_tok):
            self._absorb(conv, f)
            conv.verify_prompts += 1
            if conv.verify_prompts >= 3:
                return self._handoff(conv, "identity details not provided", STAFF)
            return "To get started, please tell me your full name and date of birth."
        name = conv.vault.value(name_tok.group(0)) or ""
        dob = _parse_dob(conv.vault.value(dob_tok.group(0)) or "")
        patient = self.clinic.verify(name, dob) if dob else None
        if not patient:
            conv.verify_failures += 1
            self.audit.append(conv.conv_id, "system", "verify_failed", "", conv.turn)
            if conv.verify_failures >= 3:
                return self._handoff(conv, "verification failed 3 times", "I wasn't able to verify your details. " + STAFF)
            return "I couldn't match those details. Please check your full name and date of birth."
        conv.patient = patient
        conv.stage = "collect"
        self.audit.append(conv.conv_id, "system", "verified", pseudonym(patient.patient_id), conv.turn)
        self._absorb(conv, f)
        return f"Thanks, {name_tok.group(0)}. " + self._collect(conv, Frame(), prompt_text)

    def _collect(self, conv: Conversation, f: Frame, prompt_text: str) -> str:
        self._absorb(conv, f)
        if conv.intent is None:
            return "Would you like to book a new appointment or cancel an existing one?"
        if conv.intent == "cancel":
            upcoming = self.clinic.upcoming(conv.patient.patient_id)
            if not upcoming:
                conv.stage = "done"
                return "I don't see any upcoming appointments to cancel."
            conv.options, conv.stage = upcoming, "offer"
            return "Which appointment would you like to cancel?\n" + _numbered(upcoming)
        if conv.department is None:
            return "Which department would you like to see - for example primary care, cardiology or dermatology?"
        options = self.clinic.search(conv.department, conv.day, conv.part)
        note = ""
        if not options:
            options = self.clinic.search(conv.department, None, conv.part) or self.clinic.search(conv.department, None, None)
            note = "There's nothing open then. "
        if not options:
            return self._handoff(conv, "no availability", "I can't find open times in that department. " + STAFF)
        conv.options, conv.stage = options, "offer"
        return f"{note}Here are the next openings in {conv.department}:\n" + _numbered(options) + "\nWhich works for you?"

    def _offer(self, conv: Conversation, f: Frame, prompt_text: str) -> str:
        pick = _pick(conv.options, f)
        if pick is None:
            if f.day or f.part or f.department:          # the patient changed the request
                conv.stage = "collect"
                return self._collect(conv, f, prompt_text)
            return "Please pick one of the options by number, or tell me another day."
        conv.pending, conv.stage = pick, "confirm"
        verb = "cancel" if conv.intent == "cancel" else "book"
        return f"Just to confirm: {verb} {pick.department} on {pick.label()}. Shall I go ahead?"

    def _confirm(self, conv: Conversation, f: Frame, prompt_text: str) -> str:
        if f.intent and f.intent != conv.intent:      # "yes, cancel that" while booking
            return "Sorry, I want to be sure. Please answer yes to go ahead, or no to choose something else."
        if f.no:
            conv.pending, conv.stage = None, "offer"
            return "No problem. Here are the options again:\n" + _numbered(conv.options)
        if not f.yes:
            return "Please answer yes to go ahead, or no to choose something else."
        slot, pid = conv.pending, conv.patient.patient_id
        if conv.intent == "cancel":
            ok = self.clinic.cancel(slot, pid)
            event, msg = "cancelled", f"Done - your {slot.department} appointment on {slot.label()} is cancelled."
        else:
            ok = self.clinic.book(slot, pid)
            event, msg = "booked", f"You're booked for {slot.department} on {slot.label()}. You'll get a reminder the day before."
        if not ok:                                      # someone else took it since we offered it
            conv.pending, conv.stage = None, "collect"
            return "Sorry, that time was just taken. " + self._collect(conv, Frame(), prompt_text)
        conv.stage = "done"
        self.audit.append(conv.conv_id, "system", event, f"{slot.department}", conv.turn)
        return msg

    # --- helpers -------------------------------------------------------------------------
    def _absorb(self, conv: Conversation, f: Frame) -> None:
        conv.intent = f.intent or conv.intent or ("book" if f.department else None)
        conv.department = f.department or conv.department
        conv.day = f.day or conv.day
        conv.part = f.part or conv.part

    def _handoff(self, conv: Conversation, reason: str, message: str) -> str:
        conv.stage, conv.handoff_reason = "handoff", reason
        self.audit.append(conv.conv_id, "system", "handoff", reason, conv.turn)
        return message


def _numbered(slots: list[Slot]) -> str:
    return "\n".join(f"  {i + 1}. {s.department}, {s.label()}" for i, s in enumerate(slots))


def _pick(options: list[Slot], f: Frame) -> Slot | None:
    if f.time:                                       # a named time wins over an option number
        hits = [s for s in options if (s.start.hour, s.start.minute) == f.time]
        return hits[0] if len(hits) == 1 else None
    if f.choice is not None and 0 <= f.choice < len(options):
        return options[f.choice]
    return None


def _parse_dob(value: str) -> dt.date | None:
    v = re.sub(r"\s+", " ", value.strip().lower().replace(",", "").replace(" of ", " ").replace(".", "/"))
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y", "%m-%d-%y",
                "%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y"):
        try:
            d = dt.datetime.strptime(re.sub(r"(\d)(st|nd|rd|th)", r"\1", v), fmt).date()
            return d if d <= dt.date.today() else d.replace(year=d.year - 100)   # two-digit years
        except ValueError:
            continue
    return None
