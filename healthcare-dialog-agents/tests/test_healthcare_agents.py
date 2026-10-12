"""Tests: PHI minimisation, agent behaviour and guardrails, audit integrity, routing, configs."""
from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.audit import AuditLog, pseudonym  # noqa: E402
from agent.clinic import Clinic  # noqa: E402
from agent.model_nlu import model_understand  # noqa: E402
from agent.scheduler import Conversation, SchedulingAgent  # noqa: E402
from phi.redact import Vault, redact  # noqa: E402
from routing.prefix_routing_sim import Params, simulate  # noqa: E402

RAW_PHI = ["Maria", "Lopez", "03/14/1985", "619", "555-0142", "123-45-6789", "00482913",
           "maria.l@example.com", "42 Ocean", "92101"]


# --- PHI minimisation -------------------------------------------------------------------
@pytest.mark.parametrize("text,kind", [
    ("my name is Maria Lopez", "NAME"),
    ("born on 03/14/1985", "DOB"),
    ("DOB: March 14, 1985", "DOB"),
    ("call me at (619) 555-0142", "PHONE"),
    ("SSN 123-45-6789", "SSN"),
    ("MRN: 00482913", "MRN"),
    ("email maria.l@example.com", "EMAIL"),
    ("I live at 42 Ocean View Drive", "ADDRESS"),
    ("zip code 92101", "ZIP"),
])
def test_each_identifier_type_is_redacted(text, kind):
    out = redact(text, Vault())
    assert f"[{kind}_1]" in out


def test_prompt_level_keeps_appointment_dates_log_level_removes_them():
    v = Vault()
    t = "Book me for March 22 at 10am"
    assert "March 22" in redact(t, v, "prompt")
    assert "March 22" not in redact(t, v, "log")


def test_vault_restores_and_reuses_tokens():
    v = Vault()
    a = redact("my name is Maria Lopez", v)
    b = redact("Maria Lopez again", v, known_names=("Maria Lopez",))
    assert a.endswith("[NAME_1]") and b.startswith("[NAME_1]")
    assert v.restore("Thanks [NAME_1]") == "Thanks Maria Lopez"


def test_known_name_does_not_break_email():
    out = redact("email maria.l@example.com", Vault(), known_names=("Maria",))
    assert out == "email [EMAIL_1]"


def test_lowercase_phrases_are_not_names():
    assert redact("i am not sure about tuesday", Vault()) == "i am not sure about tuesday"


# --- agent behaviour -----------------------------------------------------------------------
def run(turns, agent=None):
    agent = agent or SchedulingAgent(Clinic.demo())
    conv = Conversation("t")
    replies = [agent.handle(conv, m) for m in turns]
    return agent, conv, replies


def test_happy_path_books_after_explicit_yes():
    agent, conv, replies = run(["I need a cardiology appointment next Tuesday afternoon",
                                "My name is Maria Lopez, born on 03/14/1985",
                                "the first one", "yes"])
    assert conv.stage == "done" and "booked" in replies[-1]
    assert sum(1 for p in agent.clinic.booked.values() if p == "P001") == 2      # existing + new


def test_nothing_is_booked_without_yes():
    agent, conv, _ = run(["I need a cardiology appointment next Tuesday afternoon",
                          "My name is Maria Lopez, born on 03/14/1985", "the first one", "no"])
    assert conv.stage == "offer"
    assert sum(1 for p in agent.clinic.booked.values() if p == "P001") == 1


def test_slot_taken_between_offer_and_confirm_is_handled():
    agent = SchedulingAgent(Clinic.demo())
    conv = Conversation("race")
    for m in ["I need a cardiology appointment next Tuesday afternoon",
              "My name is Maria Lopez, born on 03/14/1985", "the first one"]:
        agent.handle(conv, m)
    taken = conv.pending
    agent.clinic.book(taken, "P002")                         # someone else takes it
    reply = agent.handle(conv, "yes")
    assert "just taken" in reply and conv.stage == "offer"
    assert agent.clinic.booked[taken] == "P002"              # not double-booked
    assert taken not in conv.options                         # fresh options offered


def test_cancel_flow():
    agent, conv, replies = run(["My name is Maria Lopez, DOB 03/14/1985 - I want to cancel my appointment",
                                "the first one", "yes"])
    assert conv.stage == "done" and "cancelled" in replies[-1]
    assert not agent.clinic.upcoming("P001")


def test_three_failed_verifications_hand_off():
    _, conv, _ = run(["My name is Maria Lopez, born on 01/01/1990"] * 3)
    assert conv.stage == "handoff" and conv.handoff_reason == "verification failed 3 times"


@pytest.mark.parametrize("msg", ["I have chest pain", "I can't breathe properly", "I want to kill myself"])
def test_urgent_symptoms_hand_off_on_first_turn(msg):
    _, conv, replies = run([msg])
    assert conv.stage == "handoff" and "911" in replies[0]


def test_urgent_symptom_mid_conversation_overrides_flow():
    _, conv, replies = run(["My name is Maria Lopez, born on 03/14/1985 - book cardiology",
                            "actually my chest pain is getting worse"])
    assert conv.stage == "handoff" and "911" in replies[-1]


def test_human_request_hands_off():
    _, conv, _ = run(["can I talk to a real person"])
    assert conv.stage == "handoff" and conv.handoff_reason == "patient asked for a person"


def test_two_stalled_turns_hand_off():
    _, conv, _ = run(["My name is Maria Lopez, born on 03/14/1985",
                      "hmm", "banana"])
    assert conv.stage == "handoff" and conv.handoff_reason == "not understood twice"


def test_language_layer_and_logs_never_see_raw_identifiers():
    agent, _, _ = run(["I need a cardiology appointment next Tuesday afternoon",
                       "My name is Maria Lopez, born on 03/14/1985, call me at (619) 555-0142",
                       "the first one", "yes"])
    seen = " ".join(agent.model_inputs)
    logged = json.dumps(agent.audit.entries)
    for raw in ("Maria", "Lopez", "03/14/1985", "555-0142", "P001"):
        assert raw not in seen and raw not in logged
    assert pseudonym("P001") in logged


# --- model-based understanding ---------------------------------------------------------------
def test_model_nlu_sees_only_redacted_text_and_output_is_validated():
    prompts = []

    def fake_model(prompt: str) -> str:
        prompts.append(prompt)
        if "[NAME_" in prompt and "cardio" not in prompt:
            return json.dumps({"intent": None, "department": None, "date": None, "part": None,
                               "choice": None, "yes": False, "no": False, "human": False})
        return json.dumps({"intent": "book", "department": "neurosurgery",     # not offered -> dropped
                           "date": "2026-03-17", "part": "afternoon", "choice": None,
                           "yes": False, "no": False, "human": False})

    agent = SchedulingAgent(Clinic.demo(), nlu=model_understand(fake_model))
    conv = Conversation("m")
    agent.handle(conv, "I need cardiology next Tuesday afternoon")
    agent.handle(conv, "My name is Maria Lopez, born on 03/14/1985")
    assert all("Maria" not in p and "03/14/1985" not in p for p in prompts)
    assert conv.department is None                         # the invalid department was rejected
    assert conv.part == "afternoon"


def test_model_nlu_rejects_malformed_output_and_keeps_safety_local():
    nlu = model_understand(lambda p: "not json")
    import datetime as dt
    f = nlu("I have chest pain", dt.date(2026, 3, 16))
    assert f.urgent and f.intent is None


# --- audit integrity ---------------------------------------------------------------------------
def test_audit_chain_detects_tampering():
    log = AuditLog()
    for i in range(5):
        log.append("c", "agent", "e", f"text {i}")
    assert log.verify() is None
    log.entries[2]["text"] = "edited"
    assert log.verify() == 2
    log2 = AuditLog()
    for i in range(3):
        log2.append("c", "agent", "e", f"t{i}")
    del log2.entries[1]
    assert log2.verify() == 1


# --- routing simulation -------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sims():
    out = {}
    for load, cache in ((1.0, 400_000), (1.5, 400_000), (2.0, 400_000), (0.5, 40_000)):
        p = Params(load=load, cache_tokens=cache)
        out[(load, cache)] = {pol: simulate(p, pol) for pol in ("round_robin", "least_loaded", "sticky", "kv_aware")}
    return out


def test_cache_affine_routing_cuts_median_ttft(sims):
    s = sims[(1.0, 400_000)]
    assert s["kv_aware"]["ttft_ms"]["p50"] * 5 < s["round_robin"]["ttft_ms"]["p50"]
    assert s["sticky"]["history_tokens_reused"] > s["round_robin"]["history_tokens_reused"]


def test_cache_affinity_raises_capacity(sims):
    s = sims[(1.5, 400_000)]
    assert s["round_robin"]["ttft_ms"]["p50"] > 1000 and s["least_loaded"]["ttft_ms"]["p50"] > 1000
    assert s["sticky"]["ttft_ms"]["p50"] < 50 and s["kv_aware"]["ttft_ms"]["p50"] < 50


def test_sticky_best_tail_until_imbalance_then_kv_aware(sims):
    assert sims[(1.0, 400_000)]["sticky"]["ttft_ms"]["p99"] < sims[(1.0, 400_000)]["kv_aware"]["ttft_ms"]["p99"]
    s = sims[(2.0, 400_000)]
    assert s["kv_aware"]["ttft_ms"]["p99"] < min(s[p]["ttft_ms"]["p99"] for p in ("round_robin", "least_loaded", "sticky"))


def test_pure_affinity_hurts_when_cache_is_small(sims):
    s = sims[(0.5, 40_000)]
    assert s["sticky"]["ttft_ms"]["p99"] > 2 * s["kv_aware"]["ttft_ms"]["p99"]


def test_simulation_is_deterministic():
    p = Params(load=0.3, duration_s=600)
    assert simulate(p, "kv_aware") == simulate(p, "kv_aware")


# --- serving configs -------------------------------------------------------------------------------
def test_networkpolicies_lock_down_both_phi_namespaces():
    docs = list(yaml.safe_load_all((ROOT / "serving/k8s/networkpolicy.yaml").read_text()))
    for ns in ("phi-inference", "scheduling-agent"):
        deny = next(d for d in docs if d["metadata"]["name"] == "default-deny-all" and d["metadata"]["namespace"] == ns)
        assert deny["spec"]["podSelector"] == {} and set(deny["spec"]["policyTypes"]) == {"Ingress", "Egress"}
    egress = [r for d in docs for r in d["spec"].get("egress", [])]
    assert not any("ipBlock" in t for r in egress for t in r["to"])           # no internet ranges
    llm_egress = [r for d in docs if d["metadata"]["namespace"] == "phi-inference" for r in d["spec"].get("egress", [])]
    assert {p["port"] for r in llm_egress for p in r["ports"]} == {53}         # LLM reaches only DNS


def test_audit_key_fails_closed_outside_dev(monkeypatch):
    monkeypatch.delenv("AUDIT_PSEUDONYM_KEY", raising=False)
    monkeypatch.setenv("APP_ENV", "prod")
    with pytest.raises(RuntimeError):
        pseudonym("P001")


def test_nim_manifest_enables_kv_reuse_and_runs_non_root():
    d = yaml.safe_load((ROOT / "serving/k8s/llm-nim.yaml").read_text())
    spec = d["spec"]["template"]["spec"]
    env = {e["name"]: e["value"] for e in spec["containers"][0]["env"]}
    assert env["NIM_ENABLE_KV_CACHE_REUSE"] == "1" and spec["securityContext"]["runAsNonRoot"] is True


# --- regressions from the independent review ---------------------------------------------
def _to_confirm():
    agent = SchedulingAgent(Clinic.demo())
    conv = Conversation("r")
    for m in ["I need a cardiology appointment next Tuesday afternoon",
              "My name is Maria Lopez, born on 03/14/1985", "the first one"]:
        agent.handle(conv, m)
    assert conv.stage == "confirm"
    return agent, conv


@pytest.mark.parametrize("reply", ["yes no", "that's not correct", "don't book it", "dont book it",
                                   "I'm not sure that works", "yes, cancel that", "wait", "hmm"])
def test_ambiguous_or_negated_replies_never_book(reply):
    agent, conv = _to_confirm()
    agent.handle(conv, reply)
    assert conv.stage != "done"
    assert sum(1 for p in agent.clinic.booked.values() if p == "P001") == 1


@pytest.mark.parametrize("msg", ["I cant breathe", "I can’t breathe", "I think I'm having a heart attack",
                                 "I want to die", "my chest pains are back", "she passed out"])
def test_urgent_variants(msg):
    _, conv, replies = run([msg])
    assert conv.stage == "handoff" and "911" in replies[0]


def test_urgent_after_conversation_ended_still_gets_emergency_guidance():
    agent, conv, _ = run(["I need a cardiology appointment next Tuesday afternoon",
                          "My name is Maria Lopez, born on 03/14/1985", "the first one", "yes",
                          "I have chest pains now"])
    assert conv.handoff_reason == "urgent symptoms"


def test_urgent_message_is_not_sent_to_the_language_layer():
    agent, _, _ = run(["My name is Maria Lopez, born on 03/14/1985", "I have chest pain"])
    assert not any("chest" in t for t in agent.model_inputs)


def test_named_time_beats_the_word_one():
    agent, conv, _ = run(["I need a cardiology appointment next Tuesday afternoon",
                          "My name is Maria Lopez, born on 03/14/1985",
                          "can I get the 1:30 one"])
    assert conv.pending and (conv.pending.start.hour, conv.pending.start.minute) == (13, 30)


def test_one_in_ordinary_speech_is_not_a_choice():
    _, conv, _ = run(["I need a cardiology appointment next Tuesday afternoon",
                      "My name is Maria Lopez, born on 03/14/1985",
                      "my husband also needs one"])
    assert conv.stage == "offer" and conv.pending is None


def test_longest_department_phrase_wins():
    from agent.nlu import understand
    import datetime as dt
    assert understand("I need physical therapy", dt.date(2026, 3, 16)).department == "physical therapy"


def test_insurance_name_is_not_a_request_for_a_person():
    from agent.nlu import understand
    import datetime as dt
    assert not understand("I have Humana insurance", dt.date(2026, 3, 16)).human


def test_repeated_missing_identity_hands_off():
    _, conv, _ = run(["hmm"] * 3)
    assert conv.stage == "handoff" and conv.handoff_reason == "identity details not provided"


@pytest.mark.parametrize("text,leak", [
    ("Maria Lopez 03/14/1985 cardiology", ["Maria", "Lopez", "03/14/1985"]),
    ("hi it's maria lopez, born 1985-03-14", ["maria", "lopez", "1985-03-14"]),
    ("Maria Lopez, born on 14 March 1985", ["Maria", "14 March 1985"]),
    ("Maria Lopez DOB 3.14.85", ["Maria", "3.14.85"]),
])
def test_verify_stage_identity_formats_are_redacted_and_verified(text, leak):
    agent, conv, _ = run([text])
    seen = " ".join(agent.model_inputs) + json.dumps(agent.audit.entries)
    assert not any(x in seen for x in leak)
    assert conv.patient is not None and conv.patient.patient_id == "P001"


@pytest.mark.parametrize("text,raw", [
    ("SSN 123456789", "123456789"), ("ssn 123 45 6789", "123 45 6789"),
    ("MRN# A00482913", "A00482913"), ("MRN 4829", "4829"),
    ("I live at 42 ocean view dr", "ocean view"), ("1200 N. Harbor Dr, Unit 5", "Unit 5"),
    ("call +1 619 555 0142", "555 0142"), ("6195550142", "6195550142"),
    ("my name is Sean O'Brien", "Brien"), ("I am Angus McDonald", "McDonald"),
    ("my name is José Núñez", "Núñez"), ("this is Mary-Ann Smith", "Ann"),
])
def test_more_identifier_formats(text, raw):
    assert raw not in redact(text, Vault())


def test_full_booking_leaks_none_of_the_fixture_identifiers():
    agent, _, _ = run(["I need a cardiology appointment next Tuesday afternoon",
                       "My name is Maria Lopez, born on 03/14/1985, call me at (619) 555-0142",
                       "my email is maria.l@example.com, SSN 123-45-6789, MRN 00482913, "
                       "I live at 42 Ocean View Drive, zip 92101",
                       "the first one", "yes"])
    seen = " ".join(agent.model_inputs) + json.dumps(agent.audit.entries)
    for raw in RAW_PHI + ["P001"]:
        assert raw not in seen, raw
