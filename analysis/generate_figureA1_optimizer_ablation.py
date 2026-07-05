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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    for opt in optimizers:
        ax1.plot(results[opt]["step"], results[opt]["val_acc"], label=opt, color=colors[opt])
        ax2.plot(results[opt]["step"], results[opt]["lambda_max"], label=opt, color=colors[opt])

    ax1.set_ylabel("Validation accuracy")
    ax1.set_title("Validation accuracy by optimizer")
    ax1.legend(loc="center right", frameon=False)

    ax2.set_ylabel(r"$\lambda_{max}$")
    ax2.set_xlabel("Optimization step")
    ax2.set_title(r"$\lambda_{max}$ by optimizer (coupled vs. decoupled weight decay)")
    ax2.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    fig.savefig(args.output.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved: {args.output} and {args.output.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
