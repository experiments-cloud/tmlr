"""Figure 4: distribution of the SLQ top Ritz value across seeds, by density and sampled step.
Usage: python3 -I generate_figure4_slq_boxplots.py <dir with the 36 *_spectral_density.json> <out.png>
Panel labels describe the sampled step only (early / intermediate / late); no transition is implied
for the 10% condition, where no run generalizes.
"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D, OUT = sys.argv[1], sys.argv[2]
ratios, steps, seeds = (50, 25, 10), (500, 2000, 24500), (42, 1, 2, 3)
tags = {500: "early", 2000: "intermediate", 24500: "late"}
cols = ["#1f77b4", "#ff7f0e", "#2ca02c"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
rng = np.random.default_rng(0)
for ax, st in zip(axes, steps):
    data = []
    for r in ratios:
        v = [max(json.load(open(f"{D}/checkpoints_ratio{r}_seed{s}_model_step_{st}_spectral_density.json"))["ritz_values"]) for s in seeds]
        data.append(v)
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=False,
                    medianprops=dict(color="#ff7f0e"), whiskerprops=dict(color="black"), capprops=dict(color="black"))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.35); patch.set_edgecolor("gray")
    for i, v in enumerate(data, 1):
        ax.scatter(i + rng.uniform(-0.04, 0.04, len(v)), v, color="black", zorder=3, s=28)
    ax.set_xticklabels([f"{r}%" for r in ratios]); ax.set_xlabel("Data density")
    ax.set_title(f"Step {st:,}\n({tags[st]})")
axes[0].set_ylabel(r"$\lambda_{max}$ (SLQ, 4 seeds)")
fig.suptitle("Distribution of the SLQ-estimated dominant eigenvalue across seeds, by sampled step")
fig.tight_layout(); fig.savefig(OUT, dpi=250)
