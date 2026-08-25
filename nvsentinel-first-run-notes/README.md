# Running NVSentinel Without a GPU: First-Run Notes
 
**Notes from running [NVIDIA NVSentinel](https://github.com/NVIDIA/NVSentinel)'s local
fault-injection demo (v0.6.0) on an Apple Silicon MacBook Air — what the pipeline looks
like from the inside, and the five things that only bite the first person through the door.**
 
> ⚠️ **Honest framing:** This is a first-run experience report, not a study. One laptop,
> one hour, one demo — no fleet, no GPUs, no benchmarks. The run failed to cordon under
> CPU pressure on my machine; that part is my hardware. What's worth reading is what the
> scripts reported while it was failing.
 
## What NVSentinel does
 
GPU fault detection and remediation for Kubernetes. Health monitors read DCGM, system
logs and cloud-provider maintenance events; events flow over gRPC into a central
component, get evaluated against configurable CEL rules, and a node that trips them gets
cordoned, drained and remediated — no human in the loop.
 
The [local fault-injection demo](https://github.com/NVIDIA/NVSentinel/tree/main/demos/local-fault-injection-demo)
runs the whole pipeline in a KIND cluster with a simulated DCGM, so you can watch it work
without owning a GPU.
 
## What happened
 
The cordon never occurred. Under CPU pressure — Docker reported ~1150% against 800%
available on 8 cores — MongoDB lost primary, and `fault-quarantine` couldn't watch the
change stream that drives rule evaluation:
 
```
{"level":"ERROR","msg":"Failed to watch change stream","module":"fault-quarantine",
"error":"server selection error: ... ReplicaSetNoPrimary ... connect: connection refused"}
```
 
The health event was ingested and never evaluated.
 
## First-run notes (v0.6.0 demo, August 2026)
 
Things the docs don't tell you, found the hard way:
 
1. **It needs considerably more than the stated minimum.** The README says 2 cores / 4GB.
   CPU is the binding resource — not memory — and 8 cores wasn't enough to keep MongoDB's
   replica set healthy alongside the rest of the stack. Cap Docker's CPU allocation below
   your core count so the host isn't starved.
2. **Setup takes longer than documented** — ~8 minutes against an estimated 5–6, almost
   entirely pulling the fake DCGM image. Worth knowing so you don't assume it's hung.
3. **Components crash-loop on startup, and that's expected.** `platform-connectors`
   restarted 3× and `fault-quarantine` 2× waiting for MongoDB — they coordinate through
   change streams and can't start without it. Reads as breakage on a first run; isn't.
4. **The setup script's DCGM wait fails silently.** `kubectl wait` there discards both
   stdout and stderr and doesn't check the exit code, so a timeout produces no output —
   and then `success "Fake DCGM deployed and ready"` prints anyway. If the script returns
   to the prompt with nothing after "Waiting for fake DCGM to be ready...", check the pod
   yourself: `kubectl get pods -n gpu-operator`.
5. **The verify script prints a success summary regardless of the result.** It correctly
   reports `[!] Node is NOT cordoned`, prints the diagnostics — and then prints
   `Demo Complete! 🎉` with a checkmark next to "How Fault Quarantine automatically
   cordons faulty nodes", and exits 0. For a fault-detection product, that's the one
   thing you'd least want the first-run experience to do.
   Filed as [NVIDIA/NVSentinel#1647](https://github.com/NVIDIA/NVSentinel/issues/1647);
   fixed in [#1655](https://github.com/NVIDIA/NVSentinel/pull/1655), merged.
## Design decisions worth noticing
 
Reading the Helm values while waiting for pods, three choices stood out as adoption
decisions rather than engineering ones:
 
- **Everything with teeth ships disabled.** Health monitoring runs by default; fault
  quarantine, node drainer, fault remediation and janitor are all off, behind a global
  `dryRun` flag. You're asking operators to let unfamiliar software reboot production
  machines — this is how you earn that.
- **A circuit breaker caps how much of a fleet can be cordoned in a window.** Not a
  whiteboard feature; one you add after a bad rule takes down half a cluster. At scale, a
  bad rule is more dangerous than a bad GPU.
- **Quarantine policy is configurable CEL, not hardcoded logic** — so an operator encodes
  their own risk tolerance instead of inheriting NVIDIA's. A research cluster and one
  serving customer traffic have very different views on when it's acceptable to pull a node.
## Reproduce
 
```bash
brew install kubectl helm kind jq          # or install the binaries directly
git clone https://github.com/NVIDIA/NVSentinel
cd NVSentinel/demos/local-fault-injection-demo
 
./scripts/00-setup.sh                      # ~8 min
./scripts/01-show-cluster.sh
./scripts/02-inject-error.sh
./scripts/03-verify-cordon.sh
./scripts/99-cleanup.sh                    # don't skip this
```
 
Watch the transition yourself in a second terminal: `kubectl get nodes -w`
 
## Limitations
 
- One machine, one run, under-resourced. Nothing here is a claim about how NVSentinel
  behaves on real hardware at fleet scale.
- The demo enables the GPU health monitor only (`syslogHealthMonitor: false`), so this
  exercises a single source feeding the pipeline — not the multi-source pluggable-monitor
  architecture.
- Everything about component behaviour is inferred from logs and config, not from reading
  the Go source.
## Author
 
Saurabh Rai — AI infrastructure solution architecture.
GitHub: [saurabh9498](https://github.com/saurabh9498) ·