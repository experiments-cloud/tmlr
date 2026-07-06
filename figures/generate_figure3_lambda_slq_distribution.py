"""
generate_figure3_lambda_slq_distribution.py

Generates Figure 3 for Section 5.3: distribution of the SLQ-estimated dominant
eigenvalue (lambda_max) across the 4 seeds, shown separately for each of the
three training phases (step 500 / 2000 / 24500) and three data-density
conditions (50% / 25% / 10%).

Reads directly from the 36 checkpoints_ratio<R>_seed<S>_model_step_<STEP>_
spectral_density.json files produced by spectral_density_slq.py via
run_all_spectral_density.py. lambda_max for each (ratio, seed, step) is taken
as the maximum of the "ritz_values" array in that file -- no additional
computation, smoothing, or fabrication.

Usage:
    python generate_figure3_lambda_slq_distribution.py --input_dir /path/to/jsons --output figure3_lambda_slq_distribution.png
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_all(input_dir: str):
    files = glob.glob(os.path.join(input_dir, "checkpoints_ratio*_spectral_density.json"))
    if not files:
        raise FileNotFoundError(f"No spectral_density.json files found in {input_dir}")

    data = {}
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        ratio = d["train_ratio"]
        seed = d["seed"]
        step = int(d["checkpoint"].replace(".pt", "").split("step_")[1])
        lambda_max = max(d["ritz_values"])
        data.setdefault(ratio, {}).setdefault(step, {})[seed] = lambda_max

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=".")
    parser.add_argument("--output", type=str, default="figure3_lambda_slq_distribution.png")
    args = parser.parse_args()

    data = load_all(args.input_dir)

    ratios = sorted(data.keys(), reverse=True)  # 0.5, 0.25, 0.10
    steps = sorted({s for r in data.values() for s in r.keys()})  # 500, 2000, 24500
    ratio_labels = {0.5: "50%", 0.25: "25%", 0.1: "10%"}
    step_labels = {500: "Step 500\n(early)", 2000: "Step 2,000\n(transition)", 24500: "Step 24,500\n(late)"}
    colors = {0.5: "#1f77b4", 0.25: "#ff7f0e", 0.1: "#2ca02c"}

    fig, axes = plt.subplots(1, len(steps), figsize=(4.2 * len(steps), 4.5), sharey=False)

    for ax, step in zip(axes, steps):
        box_data = []
        n_seeds_per_ratio = []
        for ratio in ratios:
            seed_dict = data[ratio].get(step, {})
            vals = list(seed_dict.values())
            box_data.append(vals)
            n_seeds_per_ratio.append(len(vals))
            if len(vals) != 4:
                print(f"WARNING: ratio={ratio}, step={step} has {len(vals)} seeds (expected 4): {seed_dict}")

        bp = ax.boxplot(box_data, tick_labels=[ratio_labels[r] for r in ratios],
                         patch_artist=True, showfliers=False, widths=0.5)
        for patch, ratio in zip(bp['boxes'], ratios):
            patch.set_facecolor(colors[ratio])
            patch.set_alpha(0.35)

        for i, (ratio, vals) in enumerate(zip(ratios, box_data), start=1):
            jitter_x = np.full(len(vals), i) + np.random.default_rng(0).uniform(-0.08, 0.08, len(vals))
            ax.scatter(jitter_x, vals, color="black", zorder=5, s=22)

        ax.set_title(step_labels[step])
        ax.set_xlabel("Data density")

    axes[0].set_ylabel(r"$\lambda_{max}$ (SLQ, 4 seeds)")
    fig.suptitle("Distribution of the SLQ-estimated dominant eigenvalue across seeds, by training phase")
    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    fig.savefig(args.output.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved: {args.output} and {args.output.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
