"""
Switch clients for the circuit breaker.

AristaEapiClient talks to an Arista top-of-rack switch over eAPI (JSON-RPC over HTTPS,
endpoint /command-api, method runCmds). Withdrawing the hub's anycast prefix is a BGP
config change: remove the `network` statement so the switch stops announcing it.

DryRunClient records the commands it would send. The tests and the demo use it; nothing
in this repo has been run against a real switch.
"""
from __future__ import annotations

import base64
import json
import ssl
import urllib.request
from dataclasses import dataclass, field


def bgp_commands(asn: int, prefix: str, announce: bool) -> list[str]:
    stmt = f"network {prefix}" if announce else f"no network {prefix}"
    # Save to startup-config, so a switch reboot does not silently re-announce a hub the
    # breaker has latched out.
    return ["enable", "configure", f"router bgp {asn}", stmt, "end",
            "copy running-config startup-config"]


@dataclass
class DryRunClient:
    asn: int = 65010
    sent: list[list[str]] = field(default_factory=list)

    def withdraw(self, prefix: str) -> None:
        self.sent.append(bgp_commands(self.asn, prefix, announce=False))

    def announce(self, prefix: str) -> None:
        self.sent.append(bgp_commands(self.asn, prefix, announce=True))


@dataclass
class AristaEapiClient:
    host: str
    username: str
    password: str
    asn: int
    verify_tls: bool = True
    timeout_s: float = 3.0

    def _run(self, cmds: list[str]) -> dict:
        body = json.dumps({"jsonrpc": "2.0", "method": "runCmds", "id": "breaker",
                           "params": {"version": 1, "cmds": cmds, "format": "json"}}).encode()
        req = urllib.request.Request(f"https://{self.host}/command-api", data=body, method="POST")
        token = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
        req.add_header("Authorization", f"Basic {token}")
        req.add_header("Content-Type", "application/json")
        ctx = None if self.verify_tls else ssl._create_unverified_context()  # noqa: S323
        with urllib.request.urlopen(req, timeout=self.timeout_s, context=ctx) as resp:
            out = json.loads(resp.read())
        if "error" in out:
            raise RuntimeError(f"eAPI error: {out['error']}")
        return out

    def withdraw(self, prefix: str) -> None:
        self._run(bgp_commands(self.asn, prefix, announce=False))

    def announce(self, prefix: str) -> None:
        self._run(bgp_commands(self.asn, prefix, announce=True))
