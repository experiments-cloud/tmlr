"""
generate_figure1_accuracy.py

Generates Figure 1 for Section 5.1: validation accuracy vs. optimization step,
mean curve with +/-1 standard deviation band across 4 seeds, for each of the
three data-density conditions (50%, 25%, 10%).

Reads directly from the 12 real telemetry JSON files produced by
train_and_grok.py + hessian_topology.py (4 seeds x 3 ratios). Does not
generate, simulate, or interpolate any data -- every plotted point comes
directly from the "steps" / "val_accuracy" arrays stored in these files.

Usage:
    python generate_figure1_accuracy.py --input_dir /path/to/jsons --output figure1_accuracy.png
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
    """Loads all grokking_telemetry_ratio*_with_hessian.json files and groups them by train_ratio."""
    files = glob.glob(os.path.join(input_dir, "grokking_telemetry_ratio*_with_hessian.json"))
    if not files:
        raise FileNotFoundError(f"No telemetry files found in {input_dir}")

    data = {}
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        ratio = d["train_ratio"]
        seed = d["seed"]
        data.setdefault(ratio, {})[seed] = d

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=".",
                         help="Directory containing the 12 grokking_telemetry_ratio*_with_hessian.json files.")
    parser.add_argument("--output", type=str, default="figure1_accuracy.png")
    args = parser.parse_args()

    data = load_all(args.input_dir)

    ratios = sorted(data.keys(), reverse=True)  # 0.5, 0.25, 0.10
    colors = {0.5: "#1f77b4", 0.25: "#ff7f0e", 0.1: "#2ca02c"}
    labels = {0.5: "50% data", 0.25: "25% data", 0.1: "10% data"}

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for ratio in ratios:
        seeds = sorted(data[ratio].keys())
        if len(seeds) != 4:
            print(f"WARNING: ratio={ratio} has {len(seeds)} seeds, expected 4. Found seeds: {seeds}")

        # All runs share the same "steps" grid (EVAL_EVERY=100, MAX_STEPS=25000),
        # verified explicitly rather than assumed.
        steps_per_seed = [data[ratio][s]["steps"] for s in seeds]
        ref_steps = steps_per_seed[0]
        for s, st in zip(seeds, steps_per_seed):
            if st != ref_steps:
                raise ValueError(f"Step grids differ across seeds for ratio={ratio}, seed={s}. "
                                  f"Cannot average pointwise without realigning first.")

        acc_matrix = np.array([data[ratio][s]["val_accuracy"] for s in seeds])  # shape (4, n_steps)
        acc_mean = acc_matrix.mean(axis=0)
        acc_std = acc_matrix.std(axis=0)

        ax.plot(ref_steps, acc_mean, color=colors[ratio], label=labels[ratio], linewidth=1.8)
        ax.fill_between(ref_steps, acc_mean - acc_std, acc_mean + acc_std,
                         color=colors[ratio], alpha=0.2, linewidth=0)

    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.9, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.legend(loc="center right", frameon=False)
    ax.set_title("Validation accuracy across training (mean $\\pm$ 1 s.d., n=4 seeds)")
    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    # Also save a PDF copy, standard for LaTeX submission (TMLR style file expects vector/PDF figures)
    fig.savefig(args.output.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved: {args.output} and {args.output.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
