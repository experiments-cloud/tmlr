"""
generate_figure4_spectral_density_grid.py

Generates Figure 4 for Section 5.3: estimated Hessian eigenvalue density
(SLQ, seed 42) at each of the 3 data-density conditions x 3 training phases
(9 checkpoints total), arranged in a grid.

Reads directly from the 9 checkpoints_ratio<R>_seed42_model_step_<STEP>_
spectral_density.json files (the seed=42 subset of the same 36-file batch
used for Figure 3). The density curve is a weighted Gaussian KDE over the
pooled Ritz values from all 10 SLQ runs at that checkpoint, weighted by their
Gauss-quadrature weights -- both taken directly from the JSON, not simulated.

Usage:
    python generate_figure4_spectral_density_grid.py --input_dir /path/to/jsons --output figure4_spectral_density_grid.png
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def load_seed42(input_dir: str):
    files = glob.glob(os.path.join(input_dir, "checkpoints_ratio*_seed42_model_step_*_spectral_density.json"))
    if len(files) == 0:
        raise FileNotFoundError(f"No seed=42 spectral_density.json files found in {input_dir}")

    records = {}
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        if d["seed"] != 42:
            continue  # extra safety, in case of a naming mismatch
        ratio = d["train_ratio"]
        step = int(d["checkpoint"].replace(".pt", "").split("step_")[1])
        records[(ratio, step)] = d

    return records


def plot_density(ax, ritz_values, weights, title):
    ritz_values = np.array(ritz_values)
    weights = np.clip(np.array(weights), 1e-12, None)

    if len(ritz_values) < 2 or np.allclose(ritz_values, ritz_values[0]):
        ax.axvline(ritz_values[0] if len(ritz_values) else 0, color="crimson")
        ax.set_title(title, fontsize=10)
        ax.set_yticks([])
        return

    kde = gaussian_kde(ritz_values, weights=weights, bw_method=0.15)
    lo, hi = ritz_values.min(), ritz_values.max()
    pad = 0.1 * (hi - lo + 1e-6)
    grid = np.linspace(lo - pad, hi + pad, 400)
    density = kde(grid)

    ax.fill_between(grid, density, alpha=0.3, color="#1f77b4")
    ax.plot(grid, density, color="#1f77b4", linewidth=1.3)

    top_eigs = np.sort(ritz_values)[::-1][:5]
    for te in top_eigs:
        ax.axvline(te, color="crimson", alpha=0.35, linewidth=0.8, linestyle="--")

    ax.set_title(title, fontsize=10)
    ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=".")
    parser.add_argument("--output", type=str, default="figure4_spectral_density_grid.png")
    args = parser.parse_args()

    records = load_seed42(args.input_dir)

    ratios = sorted({r for r, s in records.keys()}, reverse=True)
    steps = sorted({s for r, s in records.keys()})
    step_labels = {500: "step 500 (early)", 2000: "step 2,000 (transition)", 24500: "step 24,500 (late)"}
    ratio_labels = {0.5: "50% data", 0.25: "25% data", 0.1: "10% data"}

    missing = [(r, s) for r in ratios for s in steps if (r, s) not in records]
    if missing:
        print(f"WARNING: missing (ratio, step) combinations for seed 42: {missing}")

    fig, axes = plt.subplots(len(ratios), len(steps), figsize=(4.2 * len(steps), 3.0 * len(ratios)))

    for i, ratio in enumerate(ratios):
        for j, step in enumerate(steps):
            ax = axes[i, j]
            key = (ratio, step)
            if key not in records:
                ax.axis("off")
                continue
            d = records[key]
            title = f"{ratio_labels[ratio]}, {step_labels[step]}"
            plot_density(ax, d["ritz_values"], d["weights"], title)

    fig.suptitle("Estimated Hessian eigenvalue density (SLQ, seed 42)\n"
                  "dashed lines: top-5 Ritz values per checkpoint", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    fig.savefig(args.output.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved: {args.output} and {args.output.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
