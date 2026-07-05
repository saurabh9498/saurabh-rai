#!/usr/bin/env python3
"""Generate the three WS0 charts from results.csv. See README.md."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")
colors = {"agg_4w": "#76b900", "3p_1d": "#1f77b4", "2p_2d": "#ff7f0e", "1p_3d": "#d62728"}
labels = {"agg_4w": "Aggregated 4W", "3p_1d": "Disagg 3P+1D",
          "2p_2d": "Disagg 2P+2D", "1p_3d": "Disagg 1P+3D"}

# Chart 1: Pareto frontier per profile (p99 TTFT vs throughput), cc=128, prefix=0.6
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, prof in zip(axes, ["chat", "rag", "agentic"]):
    sub = df[(df.profile == prof) & (df.concurrency == 128) & (df.prefix_ratio == 0.6)]
    for _, r in sub.iterrows():
        ax.scatter(r.p99_ttft_ms, r.output_throughput_tok_s, s=140,
                   color=colors[r.split], label=labels[r.split], zorder=3)
    ax.set_title(f"{prof} (ISL {int(sub.isl.iloc[0])}/OSL {int(sub.osl.iloc[0])})")
    ax.set_xlabel("p99 TTFT (ms) — lower is better")
    ax.set_ylabel("Output throughput (tok/s)")
    ax.grid(alpha=0.3)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
fig.suptitle("Throughput vs p99 TTFT tradeoff by workload profile "
             "(4-worker budget, 60% prefix reuse, cc=128)", y=1.16)
fig.tight_layout()
fig.savefig("chart1_pareto_by_profile.png", dpi=150, bbox_inches="tight")

# Chart 2: Agentic — optimal split shifts with prefix reuse (throughput at cc=128)
fig, ax = plt.subplots(figsize=(9, 5))
sub = df[(df.profile == "agentic") & (df.concurrency == 128)]
for split in ["agg_4w", "3p_1d", "2p_2d", "1p_3d"]:
    s = sub[sub.split == split].sort_values("prefix_ratio")
    ax.plot(s.prefix_ratio, s.output_throughput_tok_s, "o-", color=colors[split],
            label=labels[split], linewidth=2, markersize=8)
ax.set_xlabel("Shared-prefix ratio (prefix-cache reuse)")
ax.set_ylabel("Output throughput (tok/s)")
ax.set_title("Agentic workload: the optimal prefill/decode split shifts with prefix reuse\n"
             "(8K ISL / 128 OSL / 4 turns, cc=128, iso 4-worker budget)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("chart2_agentic_split_vs_prefix.png", dpi=150)

# Chart 3: ITL stability — p99 ITL, agg vs disagg across profiles (cc=128, prefix 0.6)
fig, ax = plt.subplots(figsize=(9, 5))
sub = df[(df.concurrency == 128) & (df.prefix_ratio == 0.6)]
profs = ["chat", "rag", "agentic"]
x = range(len(profs))
w = 0.2
for i, split in enumerate(["agg_4w", "3p_1d", "2p_2d", "1p_3d"]):
    vals = [sub[(sub.profile == p) & (sub.split == split)].p99_itl_ms.iloc[0] for p in profs]
    ax.bar([xx + i * w for xx in x], vals, w, color=colors[split], label=labels[split])
ax.set_xticks([xx + 1.5 * w for xx in x])
ax.set_xticklabels(profs)
ax.set_ylabel("p99 inter-token latency (ms) — log scale")
ax.set_yscale("log")
ax.set_title("Prefill–decode interference: p99 ITL, aggregated vs disaggregated\n"
             "(cc=128, 60% prefix reuse, iso 4-worker budget)")
ax.legend()
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig("chart3_itl_stability.png", dpi=150)
print("3 charts saved")
