"""
Four conversations through the scheduling agent, showing what the patient sees next to
what the language layer and the audit log saw.

Run:  python demo.py
"""
from __future__ import annotations

from agent.clinic import Clinic
from agent.scheduler import Conversation, SchedulingAgent

SCRIPTS = {
    "book": [
        "Hi, I need to book a cardiology appointment next Tuesday afternoon",
        "My name is Maria Lopez, born on 03/14/1985",
        "the first one",
        "yes, book it",
    ],
    "cancel": [
        "My name is Maria Lopez, DOB 03/14/1985 - I want to cancel my appointment",
        "the first one",
        "yes",
    ],
    "verification fails": [
        "My name is Maria Lopez, born on 01/01/1990, I want a dermatology appointment",
        "My name is Maria Lopez, born on 02/02/1991",
        "My name is Maria Lopez, born on 03/03/1992",
    ],
    "urgent": [
        "I'm Priya Raman, DOB 06/30/1990, and I've had chest pain since this morning",
    ],
}


def main() -> None:
    for title, turns in SCRIPTS.items():
        agent = SchedulingAgent(Clinic.demo())
        conv = Conversation(conv_id=title)
        print(f"\n===== {title} =====")
        for msg in turns:
            print(f"patient: {msg}")
            print(f"agent:   {agent.handle(conv, msg).replace(chr(10), chr(10) + '         ')}")
        print(f"-- final stage: {conv.stage}" + (f" ({conv.handoff_reason})" if conv.handoff_reason else ""))
        print("-- language layer saw:")
        for t in agent.model_inputs:
            print(f"     {t}")
        print("-- audit log (log-level redaction):")
        for e in agent.audit.entries:
            print(f"     [{e['actor']:>7}] {e['event']:<14} {e['text'][:90]!r}")
        print(f"-- audit chain intact: {agent.audit.verify() is None}")


if __name__ == "__main__":
    main()
