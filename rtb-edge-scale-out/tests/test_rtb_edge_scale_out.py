"""Breaker behaviour, context fallback, sizing numbers, and config validity."""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
for sub in ("circuit_breaker", "context", "sizing", "training", "sim"):
    sys.path.insert(0, str(ROOT / sub))

import fleet  # noqa: E402
import kv_cache as kv  # noqa: E402
import recovery_math as rm  # noqa: E402
from arista_eapi import DryRunClient, bgp_commands  # noqa: E402
from breaker import MISSING, THERMAL, Fault, LatchedBreaker, State  # noqa: E402
from context_fallback import (ContextResolver, DictStore, domain_key, encode, page_key,  # noqa: E402
                              write_domain_vector)
from webhook_server import handle_payload, make_handler  # noqa: E402

PREFIX = "203.0.113.0/24"   # documentation range (RFC 5737)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make():
    clock, sw = Clock(), DryRunClient(asn=65010)
    return LatchedBreaker(prefix=PREFIX, switch=sw, clock=clock), clock, sw


XID79 = Fault(gpu="hub3/gpu1", kind="xid", xid=79)


# --- circuit breaker ----------------------------------------------------------------
def test_raw_fault_must_persist_before_tripping():
    b, clock, sw = make()
    b.fault_firing(XID79)
    clock.t = 14
    b.fault_firing(XID79)
    assert b.state is State.CLOSED and sw.sent == []
    clock.t = 15
    b.fault_firing(XID79)
    assert b.state is State.OPEN
    assert sw.sent == [bgp_commands(65010, PREFIX, announce=False)]


def test_prepersisted_alert_trips_immediately():
    b, _, sw = make()
    b.fault_firing(XID79, already_persisted=True)
    assert b.state is State.OPEN and len(sw.sent) == 1


@pytest.mark.parametrize("xid", [13, 74, 94])     # app error, NVLink (absent here), contained ECC
def test_non_fatal_xid_is_ignored(xid):
    b, _, sw = make()
    b.fault_firing(Fault(gpu="g", kind="xid", xid=xid), already_persisted=True)
    assert b.state is State.CLOSED and sw.sent == []


def test_raw_flapping_signal_never_trips():
    b, clock, sw = make()
    for i in range(40):                                        # up 10 s, down 5 s, repeat
        clock.t = i * 15
        b.fault_firing(XID79)
        clock.t = i * 15 + 10
        b.fault_firing(XID79)
        b.fault_resolved(XID79)
    assert b.state is State.CLOSED and sw.sent == []           # never persisted 15 s


def test_missing_gpu_trips_and_blocks_cooldown():
    b, clock, _ = make()
    gone = Fault(gpu="hub3/any", kind=MISSING)
    b.fault_firing(gone, already_persisted=True)
    assert b.state is State.OPEN
    clock.t = 3 * 3600
    b.tick()
    assert b.state is State.OPEN                               # still missing: no restore


def test_state_survives_restart(tmp_path):
    path = str(tmp_path / "state.json")
    clock, sw = Clock(), DryRunClient()
    b = LatchedBreaker(prefix=PREFIX, switch=sw, clock=clock, state_path=path)
    b.fault_firing(XID79, already_persisted=True)
    clock.t = 100
    b.fault_resolved(XID79)
    b2 = LatchedBreaker(prefix=PREFIX, switch=sw, clock=clock, state_path=path)   # restarted
    assert b2.state is State.OPEN
    clock.t = 100 + 30 * 60
    b2.tick()                                                  # cooldown carried across restart
    assert b2.state is State.CLOSED and sw.sent[-1][3] == f"network {PREFIX}"


def test_latched_resolution_does_not_reannounce():
    b, clock, sw = make()
    b.fault_firing(XID79, already_persisted=True)
    clock.t = 30
    b.fault_resolved(XID79)
    b.tick()
    assert b.state is State.OPEN and len(sw.sent) == 1        # still withdrawn


def test_restore_only_after_full_cooldown():
    b, clock, sw = make()
    b.fault_firing(XID79, already_persisted=True)
    clock.t = 10
    b.fault_resolved(XID79)
    clock.t = 10 + 29 * 60
    b.tick()
    assert b.state is State.OPEN
    clock.t = 10 + 30 * 60
    b.tick()
    assert b.state is State.CLOSED
    assert sw.sent[-1] == bgp_commands(65010, PREFIX, announce=True)


def test_new_fault_during_cooldown_restarts_it():
    b, clock, _ = make()
    b.fault_firing(XID79, already_persisted=True)
    b.fault_resolved(XID79)
    clock.t = 20 * 60
    b.fault_firing(XID79, already_persisted=True)              # flaps back
    b.fault_resolved(XID79)
    clock.t = 31 * 60                                          # 31 min after first clear
    b.tick()
    assert b.state is State.OPEN                               # only 11 min since last clear
    clock.t = 50 * 60
    b.tick()
    assert b.state is State.CLOSED


def test_flapping_gpu_causes_one_withdrawal_not_many():
    b, clock, sw = make()
    for i in range(20):                                        # healthy/unhealthy every 30 s
        clock.t = i * 60
        b.fault_firing(XID79, already_persisted=True)
        clock.t = i * 60 + 30
        b.fault_resolved(XID79)
        b.tick()
    assert len(sw.sent) == 1


def test_operator_reset_refused_while_fault_active():
    b, _, _ = make()
    b.fault_firing(XID79, already_persisted=True)
    with pytest.raises(RuntimeError):
        b.operator_reset("oncall")
    b.fault_resolved(XID79)
    b.operator_reset("oncall")
    assert b.state is State.CLOSED


def test_thermal_throttle_trips():
    b, _, _ = make()
    b.fault_firing(Fault(gpu="hub2/gpu4", kind=THERMAL), already_persisted=True)
    assert b.state is State.OPEN


def _serve(breaker, token=None):
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(breaker, threading.Lock(), token))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _post(url, body, token=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_auth_and_operator_reset():
    b, _, _ = make()
    srv, base = _serve(b, token="s3cret")
    try:
        firing = {"alerts": [{"status": "firing", "labels": {"alertname": "GPUFatalXid",
                                                              "Hostname": "hub3", "gpu": "1", "xid": "79"}}]}
        assert _post(base + "/", firing)[0] == 401                         # no token
        assert _post(base + "/", firing, "s3cret") == (200, {"state": "open"})
        assert _post(base + "/reset", {"operator": "oncall"}, "s3cret")[0] == 409   # fault active
        resolved = {"alerts": [dict(firing["alerts"][0], status="resolved")]}
        _post(base + "/", resolved, "s3cret")
        assert _post(base + "/reset", {"operator": "oncall"}, "s3cret") == (200, {"state": "closed"})
    finally:
        srv.shutdown()


def test_webhook_payload_drives_breaker():
    b, _, sw = make()
    firing = {"alerts": [{"status": "firing", "labels": {"alertname": "GPUFatalXid", "Hostname": "hub3",
                                                          "gpu": "1", "xid": "79"}}]}
    handle_payload(b, firing)
    assert b.state is State.OPEN and len(sw.sent) == 1


# --- context fallback -----------------------------------------------------------------
def test_context_tiers():
    store = DictStore()
    r = ContextResolver(store)
    url = "https://www.example.com/sports/game-recap"
    assert r.resolve(url).tier == "default"
    store.set(domain_key(url), encode(np.ones(128)))
    assert r.resolve(url).tier == "domain"
    store.set(page_key(url), encode(np.full(128, 2.0)))
    res = r.resolve(url)
    assert res.tier == "page" and res.vector[0] == 2.0
    assert r.coverage() == pytest.approx({"page": 1 / 3, "domain": 1 / 3, "default": 1 / 3})
    assert domain_key("https://www.example.com/x") == "ctx:domain:example.com"
    write_domain_vector(store, "https://news.example.org/a", np.ones(128))
    assert r.resolve("https://news.example.org/b").tier == "domain"


# --- sizing -------------------------------------------------------------------------
def test_kv_cache_exact_numbers():
    assert kv.kv_bytes_per_token(kv.LLAMA3_8B) == 65_536
    assert kv.kv_bytes_per_token(kv.LLAMA3_70B) == 163_840
    assert kv.kv_pool_bytes(kv.LLAMA3_8B, 1024, 128) == 8 * 2 ** 30          # exactly 8 GiB
    assert kv.kv_pool_bytes(kv.LLAMA3_70B, 1280, 500) / 1e9 == pytest.approx(104.86, abs=0.01)


def test_tp_choice_for_70b():
    tp, plans = kv.choose_tp(kv.LLAMA3_70B, 1280, 500)
    by_tp = {p["tp"]: p for p in plans}
    assert not by_tp[1]["fits"]                       # weights + KV exceed one 96 GB card
    assert by_tp[2]["fits"] and by_tp[2]["headroom_frac"] < 0.1   # fits, but ~5% headroom
    assert tp == 4 and by_tp[4]["headroom_gb"] == pytest.approx(195.14, abs=0.01)


def test_8b_fits_one_card_with_large_headroom():
    p = kv.vram_plan(kv.LLAMA3_8B, 1024, 128, tp=1, workspace_gb_per_gpu=4.0)
    assert p["fits"] and p["headroom_frac"] > 0.75


def test_fleet_totals():
    a, b = fleet.summarize(fleet.PHASE_A), fleet.summarize(fleet.PHASE_B)
    assert (a["total_gpus"], b["total_gpus"]) == (15, 60)
    assert a["peak_qps_per_bidding_replica"] == b["peak_qps_per_bidding_replica"] == 5000


def test_recovery_math():
    r = rm.recovery(3, 2, 1.2, read_gbps=100)
    assert r["state_read_s"] == 12 and r["total_avg_min"] == 3.5 and r["total_worst_min"] == 5
    assert rm.fits_deadline(50, 60, r["total_worst_min"])
    assert not rm.fits_deadline(50, 60, rm.recovery(60, 2, 1.2, 100)["total_worst_min"])


# --- configs ------------------------------------------------------------------------
@pytest.mark.parametrize("path", ["alerting/prometheus-rules.yaml", "alerting/alertmanager.yaml"])
def test_alerting_yaml_parses(path):
    assert yaml.safe_load((ROOT / path).read_text())


@pytest.mark.skipif(shutil.which("promtool") is None, reason="promtool not installed")
def test_promtool_accepts_rules():
    r = subprocess.run(["promtool", "check", "rules", str(ROOT / "alerting/prometheus-rules.yaml")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(shutil.which("amtool") is None, reason="amtool not installed")
def test_amtool_accepts_config(tmp_path):
    token = tmp_path / "token"
    token.write_text("x")
    cfg = (ROOT / "alerting/alertmanager.yaml").read_text().replace(
        "/etc/alertmanager/secrets/breaker-token", str(token))
    (tmp_path / "am.yaml").write_text(cfg)
    r = subprocess.run(["amtool", "check-config", str(tmp_path / "am.yaml")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_breaker_alerts_also_page_a_human():
    doc = yaml.safe_load((ROOT / "alerting/alertmanager.yaml").read_text())
    routes = doc["route"]["routes"]
    assert routes[0]["continue"] is True and routes[0]["receiver"] == "circuit-breaker"
    assert any(r["receiver"] == "oncall-pager" for r in routes[1:])
    pager = next(r for r in doc["receivers"] if r["name"] == "oncall-pager")
    assert pager.get("webhook_configs")


def test_breaker_rules_use_15s_persistence():
    doc = yaml.safe_load((ROOT / "alerting/prometheus-rules.yaml").read_text())
    rules = [r for g in doc["groups"] for r in g["rules"] if r.get("labels", {}).get("route") == "breaker"]
    assert rules and all(r["for"] == "15s" for r in rules)
