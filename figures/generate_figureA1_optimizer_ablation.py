"""
generate_figureA1_optimizer_ablation.py

Regenerates Figure A.1 (Appendix A) from the 4-seed optimizer ablation replication,
in the same visual style (mean +/- 1 std band across seeds) as Figure 1 in the main text.

Reads "step", "val_acc", "lambda_max", and "weight_l2_norm" for AdamW, Adam, and SGD
from the 4 per-seed JSON files produced by the seed-parameterized
grokking_optimizer_ablation.py (optimizer_ablation_results_seed<N>.json).
No values are recomputed, smoothed, or fabricated; only mean and standard
deviation across the 4 seeds at each step are computed.

Usage:
    python generate_figureA1_optimizer_ablation.py \
        --inputs optimizer_ablation_results_seed42.json optimizer_ablation_results_seed1.json \
                 optimizer_ablation_results_seed2.json optimizer_ablation_results_seed3.json \
        --output figureA1_optimizer_ablation.png
"""

import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                         help="List of per-seed optimizer_ablation_results_seed<N>.json files.")
    parser.add_argument("--output", type=str, default="figureA1_optimizer_ablation.png")
    args = parser.parse_args()

    all_results = []
    for path in args.inputs:
        with open(path) as f:
            all_results.append(json.load(f))

    n_seeds = len(all_results)
    print(f"Aggregating {n_seeds} seeds: {args.inputs}")

    optimizers = ["AdamW", "Adam", "SGD"]
    colors = {"AdamW": "#1f77b4", "Adam": "#ff7f0e", "SGD": "#2ca02c"}
    linestyles = {"AdamW": "-", "Adam": "--", "SGD": ":"}

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    for opt in optimizers:
        # All seeds share the same step grid for a given optimizer; verified explicitly.
        steps_per_seed = [r[opt]["step"] for r in all_results]
        ref_steps = steps_per_seed[0]
        for st in steps_per_seed:
            if st != ref_steps:
                raise ValueError(f"Step grids differ across seeds for optimizer={opt}. "
                                  f"Cannot average pointwise without realigning first.")

        acc_matrix = np.array([r[opt]["val_acc"] for r in all_results])
        lmax_matrix = np.array([r[opt]["lambda_max"] for r in all_results])
        norm_matrix = np.array([r[opt]["weight_l2_norm"] for r in all_results])

        acc_mean, acc_std = acc_matrix.mean(axis=0), acc_matrix.std(axis=0)
        lmax_mean, lmax_std = lmax_matrix.mean(axis=0), lmax_matrix.std(axis=0)
        norm_mean, norm_std = norm_matrix.mean(axis=0), norm_matrix.std(axis=0)

        ax1.plot(ref_steps, acc_mean, label=opt, color=colors[opt], linestyle=linestyles[opt], linewidth=1.8)
        ax1.fill_between(ref_steps, acc_mean - acc_std, acc_mean + acc_std, color=colors[opt], alpha=0.2, linewidth=0)

        ax2.plot(ref_steps, lmax_mean, label=opt, color=colors[opt], linestyle=linestyles[opt], linewidth=1.8)
        ax2.fill_between(ref_steps, lmax_mean - lmax_std, lmax_mean + lmax_std, color=colors[opt], alpha=0.2, linewidth=0)

        ax3.plot(ref_steps, norm_mean, label=opt, color=colors[opt], linestyle=linestyles[opt], linewidth=1.8)
        ax3.fill_between(ref_steps, np.clip(norm_mean - norm_std, 1e-4, None), norm_mean + norm_std,
                          color=colors[opt], alpha=0.2, linewidth=0)

    ax1.set_ylabel("Validation accuracy")
    ax1.set_title(f"Validation accuracy by optimizer (mean $\\pm$ 1 s.d., n={n_seeds} seeds)")
    ax1.legend(loc="center right", frameon=False)

    # Symlog scale: AdamW's lambda_max spans a much wider range than Adam/SGD's,
    # which hover near zero (occasionally slightly negative). A shared linear
    # axis makes Adam and SGD visually indistinguishable from each other and
    # from zero. Symlog (linear near zero, log beyond a threshold) keeps
    # negative values visible while separating the three curves.
    ax2.set_yscale("symlog", linthresh=1.0)
    ax2.set_ylabel(r"$\lambda_{max}$ (symlog scale)")
    ax2.set_title(r"$\lambda_{max}$ by optimizer (coupled vs. decoupled weight decay)")
    ax2.legend(loc="upper right", frameon=False)
    ax2.axhline(0, color="gray", linewidth=0.7, alpha=0.5)

    # Log scale for weight norm: Adam/SGD collapse to ~0.02-0.03 while AdamW
    # stabilizes around 35-44, a >1000x difference that a linear axis cannot
    # show meaningfully.
    ax3.set_yscale("log")
    ax3.set_ylabel(r"$\|\theta\|_2$ (log scale)")
    ax3.set_xlabel("Optimization step")
    ax3.set_title(f"Total weight L2 norm by optimizer (mean $\\pm$ 1 s.d., n={n_seeds} seeds)")
    ax3.legend(loc="center right", frameon=False)

    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    fig.savefig(args.output.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved: {args.output} and {args.output.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
