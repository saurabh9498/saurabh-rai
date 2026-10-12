"""
Model-based understanding: an LLM fills the same Frame the rule-based reader fills.

The model gets redacted text only and must answer with JSON. Every field is validated
against a fixed set of allowed values before the dialog manager sees it, so a model that
returns an unexpected department, a malformed date or extra fields cannot steer the
agent - the output is dropped and the agent asks again.

`complete` is any callable prompt -> text. In production it points at an on-prem
endpoint (NVIDIA NIM or a TensorRT-LLM server, OpenAI-compatible). The tests drive it with
a stand-in that records prompts, which is how "no identifiers reach the model" is checked.
"""
from __future__ import annotations

import datetime as dt
import json
from typing import Callable

from .clinic import DEPARTMENTS
from .nlu import Frame, is_urgent

PROMPT = """You extract scheduling details from a patient message. Identifiers have been
replaced with placeholders like [NAME_1]; never try to guess them.
Today is {today}. Reply with JSON only, using exactly these keys:
{{"intent": "book"|"cancel"|null, "department": one of {departments} or null,
  "date": "YYYY-MM-DD" or null, "part": "morning"|"afternoon"|null,
  "choice": 1|2|3|null, "yes": true|false, "no": true|false, "human": true|false}}
Message: {text}"""


def model_understand(complete: Callable[[str], str]) -> Callable[[str, dt.date], Frame]:
    def nlu(text: str, today: dt.date) -> Frame:
        f = Frame()
        # Safety is never delegated to the model: urgent wording is matched directly.
        f.urgent = is_urgent(text)
        raw = complete(PROMPT.format(today=today.isoformat(), departments=sorted(DEPARTMENTS), text=text))
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return f
        if not isinstance(data, dict) or set(data) - {"intent", "department", "date", "part", "choice", "yes", "no", "human"}:
            return f
        if data.get("intent") in ("book", "cancel"):
            f.intent = data["intent"]
        if data.get("department") in DEPARTMENTS:
            f.department = data["department"]
        if data.get("part") in ("morning", "afternoon"):
            f.part = data["part"]
        if data.get("choice") in (1, 2, 3):
            f.choice = data["choice"] - 1
        try:
            d = dt.date.fromisoformat(data["date"]) if data.get("date") else None
            f.day = d if d and today < d <= today + dt.timedelta(days=90) else None
        except (TypeError, ValueError):
            pass
        f.yes, f.no, f.human = (data.get(k) is True for k in ("yes", "no", "human"))
        return f
    return nlu
