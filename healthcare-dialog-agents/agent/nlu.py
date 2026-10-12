"""
Language understanding for the scheduling agent.

Input is ALWAYS prompt-level redacted text: identifiers are placeholders like [NAME_1].
Understanding a scheduling request never needs the patient's identity, so neither this
rule-based reader nor the model-based one (agent/model_nlu.py) ever sees it.

Output is a small structured frame. The dialog manager acts only on these fields - the
language layer can misunderstand, but it cannot take an action by itself.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

from .clinic import DEPARTMENTS

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]

# Matched on normalised text (lower case, apostrophes removed), so "can't", "can’t" and
# "cant" all match "cant". Errs towards handing off; review the list with clinicians.
URGENT = ("chest pain", "chest tightness", "heart attack", "cant breathe", "cannot breathe",
          "trouble breathing", "difficulty breathing", "short of breath", "severe bleeding",
          "bleeding heavily", "wont stop bleeding", "stroke", "face drooping", "slurred speech",
          "seizure", "overdose", "passed out", "unconscious", "suicid", "kill myself", "want to die",
          "end my life", "hurt myself", "harm myself")
HUMAN = re.compile(r"\b(human being|a human|real person|representative|operator|front desk|"
                   r"speak (?:to|with) (?:someone|a person|staff)|talk (?:to|with) (?:someone|a person|staff))\b")
NEGATION = re.compile(r"\b(not|dont|do not|never|cancel that|wrong|incorrect|wait)\b")


def normalise(text: str) -> str:
    return re.sub(r"['’`]", "", text.lower())


def is_urgent(raw_text: str) -> bool:
    """Run on the RAW message, locally. Never delegated to a model, never redacted first."""
    t = normalise(raw_text)
    return any(k in t for k in URGENT)


@dataclass
class Frame:
    intent: str | None = None            # "book" | "cancel"
    department: str | None = None
    day: dt.date | None = None
    part: str | None = None              # "morning" | "afternoon"
    choice: int | None = None            # 0-based pick from offered options
    time: tuple[int, int] | None = None  # (hour, minute) if the user named a time
    yes: bool = False
    no: bool = False
    human: bool = False
    urgent: bool = False


def understand(text: str, today: dt.date) -> Frame:
    t = text.lower()
    n = normalise(text)
    f = Frame()
    f.urgent = is_urgent(text)
    f.human = bool(HUMAN.search(n))
    if re.search(r"\b(cancel|call off)\b", t):
        f.intent = "cancel"
    elif re.search(r"\b(book|schedule|appointment|see (?:a|the|my)|make an?)\b", t):
        f.intent = "book"
    # Longest phrase first, so "physical therapy" is not read as "physical" (a check-up).
    for phrase, dept in sorted(((w, d) for d, ws in DEPARTMENTS.items() for w in ws), key=lambda x: -len(x[0])):
        if re.search(rf"\b{re.escape(phrase)}\b", t):
            f.department = dept
            break
    f.day = _day(t, today)
    if re.search(r"\bmorning\b|\bbefore noon\b", t):
        f.part = "morning"
    elif re.search(r"\bafternoon\b|\bafter lunch\b", t):
        f.part = "afternoon"
    # Ordinals only - bare "one" is too common ("I need one", "the 3:30 one").
    m = re.search(r"\b(first|second|third|1st|2nd|3rd|option [123]|number [123])\b", t) \
        or re.fullmatch(r"\s*#?([123])\s*[.!]?\s*", t)
    if m:
        f.choice = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2}.get(
            m.group(1), None)
        if f.choice is None:
            f.choice = int(m.group(1)[-1]) - 1
    m = re.search(r"\b(\d{1,2}):(\d{2})\b|\b(\d{1,2})\s*(am|pm)\b", t)
    if m:
        h = int(m.group(1) or m.group(3))
        mi = int(m.group(2) or 0)
        if m.group(4) == "pm" and h < 12 or (not m.group(4) and h < 8):
            h += 12
        f.time = (h, mi)
    yes = bool(re.search(r"\b(yes|yeah|yep|yup|correct|confirm|confirmed|sounds good|that works|"
                         r"please do|go ahead|book it|do it)\b", n))
    no = bool(re.search(r"\b(no|nope|nah|different|another|other one)\b", n)) or bool(NEGATION.search(n))
    # Only an unambiguous yes counts: "yes no", "that's not correct" and "don't book it" are not yes.
    f.yes, f.no = yes and not no, no
    return f


def _day(t: str, today: dt.date) -> dt.date | None:
    if "tomorrow" in t:
        return today + dt.timedelta(days=1)
    for i, name in enumerate(WEEKDAYS):
        if re.search(rf"\b{name}\b", t):
            ahead = (i - today.weekday()) % 7 or 7          # next occurrence, never today
            return today + dt.timedelta(days=ahead)
    m = re.search(rf"\b({'|'.join(MONTHS)})\s+(\d{{1,2}})\b", t)
    if m:
        return _future(today, MONTHS.index(m.group(1)) + 1, int(m.group(2)))
    m = re.search(r"\b(\d{1,2})/(\d{1,2})\b(?!/)", t)
    if m:
        return _future(today, int(m.group(1)), int(m.group(2)))
    return None


def _future(today: dt.date, month: int, day: int) -> dt.date | None:
    try:
        d = dt.date(today.year, month, day)
        return d if d > today else dt.date(today.year + 1, month, day)
    except ValueError:
        return None
