"""
generate_figure2_measurement_noise_v2.py

Updated version of Figure 2: adds the fixed-mini-batch condition (the third
column of Table 4) as a fourth strip-plot column for each of the four
checkpoints that were re-tested with a fixed batch, alongside the original
20-iteration (fresh batch) and 100-iteration (fresh batch) conditions.

Reads directly from the real *_noise_diagnostic.json files (fresh-batch 20it,
fresh-batch 100it, and fixed-batch 20it variants). No values are simulated;
every point plotted is one of the raw lambda_max_samples stored in these files.

Usage:
    python generate_figure2_measurement_noise_v2.py --input_dir /path/to/jsons --output figure2_measurement_noise.png
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_ITERATIONS_BEFORE_FLAG_EXISTED = 20


def load_diagnostics(input_dir: str):
    """Loads all three families of noise-diagnostic files: fresh-batch (with or
    without the num_iterations/fixed_batch fields, for the earliest runs),
    and the newer fixed-batch runs which always carry both fields."""
    files = glob.glob(os.path.join(input_dir, "*noise_diagnostic.json"))
    if not files:
        raise FileNotFoundError(f"No *noise_diagnostic.json files found in {input_dir}")

    records = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)

        n_iter = d.get("num_iterations", DEFAULT_ITERATIONS_BEFORE_FLAG_EXISTED)
        fixed_batch = d.get("fixed_batch", False)

        ckpt = d["checkpoint"]
        parts = ckpt.replace(".pt", "").split("/")
        run_name = parts[-2] if len(parts) >= 2 else "unknown"
        step = parts[-1].replace("model_step_", "")

        records.append({
            "run_name": run_name,
            "ratio": d["train_ratio"],
            "step": int(step),
            "n_iter": n_iter,
            "fixed_batch": fixed_batch,
            "lambda_max_samples": d["lambda_max_samples"],
        })

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=".")
    parser.add_argument("--output", type=str, default="figure2_measurement_noise.png")
    args = parser.parse_args()

    records = load_diagnostics(args.input_dir)

    # Keep only the checkpoints/conditions actually used in Table 4 and its
    # accompanying figure: the 6 original (ratio, step) checkpoints at 20
    # iterations, the 4 that were retested at 100 iterations, and the same 4
    # retested with a fixed batch at 20 iterations.
    highlight_labels = {"ratio10_seed42_step2000", "ratio10_seed42_step24500"}

    # Sort: by (ratio desc, step asc), then within each checkpoint by condition
    # order fresh-20 -> fresh-100 -> fixed-20, matching the table's column order.
    def condition_rank(r):
        if r["fixed_batch"]:
            return 2
        return 0 if r["n_iter"] == 20 else 1

    records.sort(key=lambda r: (-r["ratio"], r["step"], condition_rank(r)))

    fig, ax = plt.subplots(figsize=(13, 5.5))

    x_positions = []
    x_labels = []
    pos = 0
    rng = np.random.default_rng(0)

    for r in records:
        vals = np.array(r["lambda_max_samples"])
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        label_key = f"{r['run_name']}_step{r['step']}"
        color = "#d62728" if label_key in highlight_labels else "#1f77b4"
        alpha = 0.9 if label_key in highlight_labels else 0.5

        ax.scatter(np.full_like(vals, pos) + jitter, vals, color=color, alpha=alpha, s=32, zorder=3)
        ax.hlines(vals.mean(), pos - 0.2, pos + 0.2, color="black", linewidth=1.5, zorder=4)

        cv = 100 * vals.std() / vals.mean() if vals.mean() != 0 else float("nan")
        ax.text(pos, vals.max() + 0.05 * (vals.max() - vals.min() + 1), f"CV={cv:.0f}%",
                ha="center", fontsize=8)

        x_positions.append(pos)
        if r["fixed_batch"]:
            cond_tag = "fixed batch, 20it"
        else:
            cond_tag = f"{r['n_iter']}it, fresh batch"
        x_labels.append(f"{int(r['ratio']*100)}%\nstep {r['step']}\n({cond_tag})")
        pos += 1

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=7.5)
    ax.set_ylabel(r"$\lambda_{max}$ (10 repeated measurements, frozen weights)")
    ax.set_title("Repeated power-iteration measurements on fixed checkpoints:\n"
                  "fresh batch (20 / 100 it.) vs. fixed batch (20 it.), highlighting the\n"
                  "resolved case (10% / step 2000) and the persistent bimodal case (10% / step 24500)")
    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    fig.savefig(args.output.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved: {args.output} and {args.output.replace('.png', '.pdf')}")
    print(f"Total columns plotted: {len(records)}")


if __name__ == "__main__":
    main()
