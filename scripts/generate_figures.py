"""Generate evaluation figures for the phase 3 report."""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

os.makedirs("reports/figures", exist_ok=True)

with open("reports/evaluation_report.json") as f:
    data = json.load(f)

# ─── Chart 1: Hybrid vs. Vector-Only Metric Comparison ───────────────────────
metrics = ["Citation\nPrecision", "Faithfulness\nRate", "No-Evidence\nAccuracy"]
hybrid_vals = [1.000, 0.950, 0.950]
vector_vals = [1.000, 1.000, 0.900]

x = np.arange(len(metrics))
width = 0.32

fig, ax = plt.subplots(figsize=(7, 4.5))
bars1 = ax.bar(x - width / 2, hybrid_vals, width, label="Hybrid (BM25+FAISS)",
               color="#2E86AB", zorder=3)
bars2 = ax.bar(x + width / 2, vector_vals, width, label="Vector-only (FAISS)",
               color="#E84855", zorder=3)

ax.set_ylabel("Score", fontsize=11)
ax.set_title("Hybrid vs. Vector-Only: Evaluation Metrics (20 queries)",
             fontsize=12, fontweight="bold", pad=10)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylim(0.85, 1.05)
ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))
ax.legend(fontsize=9, loc="lower right")
ax.grid(axis="y", alpha=0.35, zorder=0)
ax.set_facecolor("#F8F9FA")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{bar.get_height():.3f}", ha="center", va="bottom",
            fontsize=9, color="#2E86AB", fontweight="bold")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{bar.get_height():.3f}", ha="center", va="bottom",
            fontsize=9, color="#E84855", fontweight="bold")

plt.tight_layout()
plt.savefig("reports/figures/fig1_metric_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig1_metric_comparison.png")

# ─── Chart 2: Per-Query Faithfulness (Hybrid mode) ───────────────────────────
hybrid_results = data["hybrid"]
query_ids = [r["query_id"] for r in hybrid_results]
faithful = [r["faithful"] for r in hybrid_results]

colors = ["#27AE60" if f else "#E84855" for f in faithful]

fig, ax = plt.subplots(figsize=(11, 3.8))
ax.bar(range(len(query_ids)), [1] * len(query_ids), color=colors,
       zorder=3, edgecolor="white", linewidth=1.5)

ax.set_xticks(range(len(query_ids)))
ax.set_xticklabels(query_ids, rotation=0, fontsize=9)
ax.set_yticks([])
ax.set_title("Per-Query Faithfulness — Hybrid Pipeline (20 evaluation queries)",
             fontsize=12, fontweight="bold", pad=12)
ax.set_xlim(-0.6, len(query_ids) - 0.4)
ax.set_ylim(0, 1.18)

# Query-type separators and labels
ax.axvline(x=9.5, color="gray", linestyle="--", alpha=0.45)
ax.axvline(x=14.5, color="gray", linestyle="--", alpha=0.45)
ax.text(4.5, 1.10, "Direct (D01–D10)", ha="center", fontsize=9,
        style="italic", color="#555555")
ax.text(12.0, 1.10, "Synthesis (S01–S05)", ha="center", fontsize=9,
        style="italic", color="#555555")
ax.text(17.0, 1.10, "Edge (E01–E05)", ha="center", fontsize=9,
        style="italic", color="#555555")

# Annotate the one failure
fail_idx = query_ids.index("S01")
ax.text(fail_idx, 1.04, "FAIL", ha="center", fontsize=8,
        color="#E84855", fontweight="bold")

faithful_patch = mpatches.Patch(color="#27AE60", label="Faithful (PASS)")
unfaithful_patch = mpatches.Patch(color="#E84855", label="Unfaithful (FAIL)")
ax.legend(handles=[faithful_patch, unfaithful_patch], fontsize=9,
          loc="lower right")

ax.set_facecolor("#F8F9FA")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

plt.tight_layout()
plt.savefig("reports/figures/fig2_per_query_faithfulness.png", dpi=150,
            bbox_inches="tight")
plt.close()
print("Saved fig2_per_query_faithfulness.png")
