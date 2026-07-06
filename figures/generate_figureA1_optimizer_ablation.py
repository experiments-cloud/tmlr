"""
generate_figureA1_optimizer_ablation.py

Regenerates Figure A.1 (Appendix A) directly from optimizer_ablation_results.json,
in the same visual style (English labels, PDF export) as Figures 1-4 in the main text.

Reads "step", "val_acc", and "lambda_max" for AdamW, Adam, and SGD directly from
the JSON produced by the modified grokking_optimizer_ablation.py. No values are
recomputed, smoothed, or fabricated.

Usage:
    python generate_figureA1_optimizer_ablation.py --input optimizer_ablation_results.json --output figureA1_optimizer_ablation.png
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="optimizer_ablation_results.json")
    parser.add_argument("--output", type=str, default="figureA1_optimizer_ablation.png")
    args = parser.parse_args()

    with open(args.input) as f:
        results = json.load(f)

    optimizers = ["AdamW", "Adam", "SGD"]
    colors = {"AdamW": "#1f77b4", "Adam": "#ff7f0e", "SGD": "#2ca02c"}
    linestyles = {"AdamW": "-", "Adam": "--", "SGD": ":"}

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    for opt in optimizers:
        ax1.plot(results[opt]["step"], results[opt]["val_acc"], label=opt,
                  color=colors[opt], linestyle=linestyles[opt], linewidth=1.8)
        ax2.plot(results[opt]["step"], results[opt]["lambda_max"], label=opt,
                  color=colors[opt], linestyle=linestyles[opt], linewidth=1.8)
        ax3.plot(results[opt]["step"], results[opt]["weight_l2_norm"], label=opt,
                  color=colors[opt], linestyle=linestyles[opt], linewidth=1.8)

    ax1.set_ylabel("Validation accuracy")
    ax1.set_title("Validation accuracy by optimizer")
    ax1.legend(loc="center right", frameon=False)

    # Symlog scale: AdamW's lambda_max spans ~0-1440, while Adam/SGD hover in
    # [-0.04, 0.05]. A shared linear axis makes Adam and SGD visually
    # indistinguishable from each other and from zero. Symlog (linear near
    # zero, log beyond a threshold) keeps negative values visible while
    # separating the three curves at very different magnitudes.
    ax2.set_yscale("symlog", linthresh=1.0)
    ax2.set_ylabel(r"$\lambda_{max}$ (symlog scale)")
    ax2.set_title(r"$\lambda_{max}$ by optimizer (coupled vs. decoupled weight decay)")
    ax2.legend(loc="upper right", frameon=False)
    ax2.axhline(0, color="gray", linewidth=0.7, alpha=0.5)

    # Log scale for weight norm: Adam/SGD collapse to ~0.02-0.03 while AdamW
    # stabilizes around 37-42, a >1000x difference that a linear axis cannot
    # show meaningfully.
    ax3.set_yscale("log")
    ax3.set_ylabel(r"$\|\theta\|_2$ (log scale)")
    ax3.set_xlabel("Optimization step")
    ax3.set_title("Total weight L2 norm by optimizer")
    ax3.legend(loc="center right", frameon=False)

    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    fig.savefig(args.output.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved: {args.output} and {args.output.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
