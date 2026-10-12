"""
Append-only, hash-chained audit trail.

Every entry records who acted (patient, agent, system), what happened, and the LOG-level
redacted text - no identifiers, no explicit dates. Each entry's hash covers the previous
entry's hash, so editing or deleting any past entry breaks verification from that point
on. In production the chain head is periodically written to separate write-once storage.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field

GENESIS = "0" * 64
# Per-deployment secret. Patient IDs are logged as keyed pseudonyms: auditors holding the key
# can link entries to a record; anyone reading the log alone cannot. Fails closed: without
# a key, only an explicit APP_ENV=dev (the default for running this repo) uses a dev key.
def _key() -> bytes:
    key = os.environ.get("AUDIT_PSEUDONYM_KEY")
    if key:
        return key.encode()
    if os.environ.get("APP_ENV", "dev") == "dev":
        return b"dev-only-key"
    raise RuntimeError("AUDIT_PSEUDONYM_KEY must be set outside development")


def pseudonym(patient_id: str) -> str:
    return "pt_" + hmac.new(_key(), patient_id.encode(), hashlib.sha256).hexdigest()[:16]


@dataclass
class AuditLog:
    entries: list[dict] = field(default_factory=list)

    def append(self, conversation: str, actor: str, event: str, text: str = "", seq_time: float = 0.0) -> dict:
        prev = self.entries[-1]["hash"] if self.entries else GENESIS
        body = {"seq": len(self.entries), "t": seq_time, "conversation": conversation,
                "actor": actor, "event": event, "text": text, "prev": prev}
        body["hash"] = _digest(body)
        self.entries.append(body)
        return body

    def verify(self) -> int | None:
        """Index of the first entry that fails verification, or None if the chain is intact."""
        prev = GENESIS
        for i, e in enumerate(self.entries):
            body = {k: v for k, v in e.items() if k != "hash"}
            if e["prev"] != prev or _digest(body) != e["hash"]:
                return i
            prev = e["hash"]
        return None


def _digest(body: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in body.items() if k != "hash"},
                                     sort_keys=True).encode()).hexdigest()
