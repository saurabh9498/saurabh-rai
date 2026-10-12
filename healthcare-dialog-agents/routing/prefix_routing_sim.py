"""
Does routing a conversation back to the replica that holds its KV cache matter?

A multi-turn agent re-sends the whole conversation every turn: system prompt, tool
definitions, and every earlier message, plus the new one. A replica that served the
previous turn still has the key-value (KV) cache for all of that, so it only has to
prefill the new message. Any other replica must prefill the whole history.

This discrete-event simulation compares four routing policies on the same traffic:

  round_robin   next replica in turn                        ignores cache and load
  least_loaded  replica with the shortest prefill backlog   ignores cache
  sticky        hash(conversation) -> replica               ignores load
  kv_aware      minimise (backlog + uncached tokens / prefill rate)
                - the cost function KV-aware routers such as Dynamo's use

Model (stated assumptions, not measurements):
  * Time to first token (TTFT) = waiting for the replica + prefill of uncached tokens.
    Decode runs in continuous batching and is modelled only as the gap before the
    patient's next turn.
  * The shared system prompt is cached once per replica (a pinned prefix, never evicted)
    after the replica's first request.
  * Each replica holds a finite KV cache (in tokens) for conversation history, with
    least-recently-used eviction. The pinned system prompt does not count against it.
  * `load` is the prefill utilisation per replica if every turn had to prefill its whole
    history (system prompt already cached). Cache reuse lowers the real utilisation.
  * Conversations arrive as a Poisson process; patients think for an exponential time.

Results depend on these parameters - run the sweep, read the shape, then measure on real
hardware. Run:  python routing/prefix_routing_sim.py [--out results/routing.json]
"""
from __future__ import annotations

import argparse
import heapq
import json
import random
from collections import OrderedDict
from dataclasses import asdict, dataclass

POLICIES = ("round_robin", "least_loaded", "sticky", "kv_aware")


@dataclass(frozen=True)
class Params:
    replicas: int = 4
    prefill_tps: float = 8_000.0          # prefill tokens/s per replica (assumption)
    system_tokens: int = 1_500            # system prompt + tool definitions
    user_tokens: int = 60
    reply_tokens: int = 120
    turns: int = 8
    think_s: float = 20.0
    decode_tps: float = 40.0              # per-stream decode speed: sets reply duration
    cache_tokens: int = 400_000           # KV cache per replica, in tokens
    load: float = 0.7                     # offered prefill load with NO cache reuse, per replica
    duration_s: float = 3_600.0
    seed: int = 0

    def prompt(self, turn: int) -> int:   # turn is 1-based
        return self.system_tokens + (turn - 1) * (self.user_tokens + self.reply_tokens) + self.user_tokens

    def conv_rate(self) -> float:
        """Conversations/s giving `load` utilisation if no history were ever cached."""
        work = sum(self.prompt(k) - self.system_tokens for k in range(1, self.turns + 1)) / self.prefill_tps
        return self.load * self.replicas / work


class Replica:
    def __init__(self, p: Params):
        self.p = p
        self.free_at = 0.0
        self.cache: OrderedDict[int, int] = OrderedDict()     # conversation -> cached tokens
        self.used = 0
        self.has_system = False

    def uncached(self, conv: int, prompt: int) -> int:
        sys_part = 0 if self.has_system else self.p.system_tokens
        history = prompt - self.p.system_tokens
        return sys_part + max(history - self.cache.get(conv, 0), 0)

    def store(self, conv: int, history_tokens: int) -> None:
        """Cache this conversation's history (everything after the shared system prompt)."""
        if conv in self.cache:
            self.used -= self.cache.pop(conv)
        self.cache[conv] = history_tokens
        self.used += history_tokens
        while self.used > self.p.cache_tokens and len(self.cache) > 1:
            _, t = self.cache.popitem(last=False)
            self.used -= t
        self.has_system = True


def simulate(p: Params, policy: str) -> dict:
    rng = random.Random(p.seed)
    reps = [Replica(p) for _ in range(p.replicas)]
    rr = 0
    events: list[tuple[float, int, int]] = []          # (time, conversation, turn)
    t, cid = 0.0, 0
    rate = p.conv_rate()
    while True:
        t += rng.expovariate(rate)
        if t > p.duration_s:
            break
        heapq.heappush(events, (t, cid, 1))
        cid += 1
    sticky = {c: random.Random(c * 7919 + p.seed).randrange(p.replicas) for c in range(cid)}

    ttfts, history_tok, history_computed = [], 0, 0
    while events:
        now, conv, turn = heapq.heappop(events)
        prompt = p.prompt(turn)
        if policy == "round_robin":
            r = rr % p.replicas
            rr += 1
        elif policy == "least_loaded":
            r = min(range(p.replicas), key=lambda i: (reps[i].free_at, i))
        elif policy == "sticky":
            r = sticky[conv]
        else:
            r = min(range(p.replicas), key=lambda i: (max(reps[i].free_at - now, 0)
                                                      + reps[i].uncached(conv, prompt) / p.prefill_tps, i))
        rep = reps[r]
        sys_cold = 0 if rep.has_system else p.system_tokens
        work = rep.uncached(conv, prompt)
        start = max(now, rep.free_at)
        done = start + work / p.prefill_tps
        rep.free_at = done
        rep.store(conv, prompt - p.system_tokens + p.reply_tokens)
        ttfts.append(done - now)
        history_tok += prompt - p.system_tokens
        history_computed += work - sys_cold
        if turn < p.turns:
            nxt = done + p.reply_tokens / p.decode_tps + rng.expovariate(1 / p.think_s)
            heapq.heappush(events, (nxt, conv, turn + 1))

    ttfts.sort()
    pct = lambda q: ttfts[min(int(q * len(ttfts)), len(ttfts) - 1)] * 1000   # noqa: E731
    return {"policy": policy, "requests": len(ttfts),
            "ttft_ms": {"p50": round(pct(0.50), 1), "p90": round(pct(0.90), 1), "p99": round(pct(0.99), 1),
                        "mean": round(sum(ttfts) / len(ttfts) * 1000, 1)},
            "history_tokens_reused": round(1 - history_computed / history_tok, 3)}


def sweep(base: Params) -> list[dict]:
    rows = []
    for load in (0.5, 1.0, 1.5, 2.0):
        for cache in (400_000, 40_000):
            p = Params(**{**asdict(base), "load": load, "cache_tokens": cache})
            for pol in POLICIES:
                rows.append({"load": load, "cache_tokens": cache, **simulate(p, pol)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    base = Params()
    rows = sweep(base)
    print(f"{'load':>5} {'cache':>8} {'policy':<13} {'p50':>8} {'p99':>9} {'history reused':>15}")
    for r in rows:
        print(f"{r['load']:>5} {r['cache_tokens']:>8} {r['policy']:<13} {r['ttft_ms']['p50']:>8} "
              f"{r['ttft_ms']['p99']:>9} {r['history_tokens_reused']:>15}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"assumptions": asdict(base), "rows": rows}, fh, indent=2)
            fh.write("\n")


if __name__ == "__main__":
    main()
