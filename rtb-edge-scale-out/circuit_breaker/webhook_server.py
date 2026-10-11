"""
Alertmanager webhook receiver that drives the latched breaker.

Alertmanager POSTs a JSON payload with a list of alerts, each "firing" or "resolved".
Labels set by the Prometheus rules (alerting/prometheus-rules.yaml) say which GPU and
which fault. A background thread calls breaker.tick() so the cooldown can complete.

Endpoints:
    POST /       Alertmanager webhook
    POST /reset  operator override, body {"operator": "name"}; refused while faults are active
    GET  /       current state and recent events

If BREAKER_TOKEN is set, every POST must carry "Authorization: Bearer <token>"; otherwise
any pod that can reach the service could withdraw the hub. Alertmanager sends it via
http_config.authorization (alerting/alertmanager.yaml).

Run locally (dry run, no switch):
    python circuit_breaker/webhook_server.py --prefix 203.0.113.0/24 --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arista_eapi import AristaEapiClient, DryRunClient  # noqa: E402
from breaker import MISSING, THERMAL, Fault, LatchedBreaker  # noqa: E402

log = logging.getLogger("webhook")


def fault_from_alert(alert: dict) -> Fault:
    labels = alert.get("labels", {})
    gpu = f"{labels.get('Hostname', '?')}/gpu{labels.get('gpu', '?')}"
    if labels.get("alertname") == "GPUSustainedThermalThrottle":
        return Fault(gpu=gpu, kind=THERMAL)
    if labels.get("alertname") in ("GPUMissing", "DCGMExporterDown"):
        return Fault(gpu=f"{labels.get('Hostname', labels.get('instance', '?'))}/any", kind=MISSING)
    return Fault(gpu=gpu, kind="xid", xid=int(float(labels.get("xid", "0"))))


def handle_payload(breaker: LatchedBreaker, payload: dict) -> None:
    for alert in payload.get("alerts", []):
        fault = fault_from_alert(alert)
        if alert.get("status") == "firing":
            breaker.fault_firing(fault, already_persisted=True)   # rule's `for: 15s` already applied
        else:
            breaker.fault_resolved(fault)


def make_handler(breaker: LatchedBreaker, lock: threading.Lock, token: str | None = None):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code: int, body: dict) -> None:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def do_POST(self):  # noqa: N802
            if token and self.headers.get("Authorization") != f"Bearer {token}":
                return self._reply(401, {"error": "unauthorized"})
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            with lock:
                if self.path == "/reset":
                    try:
                        breaker.operator_reset(payload.get("operator", "unknown"))
                    except RuntimeError as e:
                        return self._reply(409, {"error": str(e), "state": breaker.state.value})
                else:
                    handle_payload(breaker, payload)
                state = breaker.state.value
            self._reply(200, {"state": state})

        def do_GET(self):  # noqa: N802  (status for operators)
            with lock:
                body = {"state": breaker.state.value, "events": breaker.events[-20:]}
            self._reply(200, body)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--port", type=int, default=9099)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.dry_run:
        switch = DryRunClient()
    else:
        switch = AristaEapiClient(host=os.environ["EAPI_HOST"], username=os.environ["EAPI_USER"],
                                  password=os.environ["EAPI_PASSWORD"], asn=int(os.environ["BGP_ASN"]))
    breaker = LatchedBreaker(prefix=args.prefix, switch=switch,
                             state_path=os.environ.get("BREAKER_STATE", "/var/lib/breaker/state.json"))
    lock = threading.Lock()

    def ticker():
        while True:
            time.sleep(10)
            try:
                with lock:
                    breaker.tick()
            except Exception:  # a failed re-announce must not kill the loop; retry next tick
                log.exception("tick failed; will retry")

    threading.Thread(target=ticker, daemon=True).start()
    handler = make_handler(breaker, lock, token=os.environ.get("BREAKER_TOKEN"))
    ThreadingHTTPServer(("0.0.0.0", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
