"""
generate_figure2_measurement_noise.py

Generates Figure 2 for Section 5.2: individual repeated lambda_max measurements
(strip plot) on the SAME frozen checkpoint weights, comparing the 20-iteration
and 100-iteration power-iteration budgets, for the checkpoints in Table 4.

Reads directly from the *_noise_diagnostic.json files produced by
measure_intrinsic_noise.py. Does not simulate, interpolate, or fabricate any
values -- every point plotted is one of the 10 raw lambda_max_samples stored
in these files.

Iteration budget handling: the six checkpoints run before the --num_iterations
flag was added to measure_intrinsic_noise.py do not have a "num_iterations"
field in their JSON. For those files we assume the then-current script default
of 20 iterations (documented explicitly here and printed as a warning), rather
than silently guessing. Files with "iters100" in their filename are labeled
100 iterations directly from the "num_iterations" field they do carry.

Usage:
    python generate_figure2_measurement_noise.py --input_dir /path/to/jsons --output figure2_measurement_noise.png
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_ITERATIONS_BEFORE_FLAG_EXISTED = 20  # documented assumption, see module docstring


def load_diagnostics(input_dir: str):
    files = glob.glob(os.path.join(input_dir, "*noise_diagnostic.json"))
    if not files:
        raise FileNotFoundError(f"No *noise_diagnostic.json files found in {input_dir}")

    records = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)

        if "num_iterations" in d:
            n_iter = d["num_iterations"]
            assumed = False
        else:
            n_iter = DEFAULT_ITERATIONS_BEFORE_FLAG_EXISTED
            assumed = True

        ckpt = d["checkpoint"]
        # e.g. "checkpoints/ratio10_seed42/model_step_24500.pt" -> "ratio10_seed42_step24500"
        parts = ckpt.replace(".pt", "").split("/")
        run_name = parts[-2] if len(parts) >= 2 else "unknown"
        step = parts[-1].replace("model_step_", "")
        label = f"{run_name}_step{step}"

        records.append({
            "label": label,
            "ratio": d["train_ratio"],
            "step": int(step),
            "n_iter": n_iter,
            "assumed_iterations": assumed,
            "lambda_max_samples": d["lambda_max_samples"],
        })

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=".")
    parser.add_argument("--output", type=str, default="figure2_measurement_noise.png")
    args = parser.parse_args()

    records = load_diagnostics(args.input_dir)

    for r in records:
        if r["assumed_iterations"]:
            print(f"NOTE: {r['label']} has no 'num_iterations' field; "
                  f"assuming the pre-flag script default of {DEFAULT_ITERATIONS_BEFORE_FLAG_EXISTED}.")

    # We highlight the two checkpoints discussed in the text: the resolved case
    # (10% data, step 2000) and the persistent bimodal case (10% data, step 24500).
    # All other checkpoints are shown for completeness but not individually annotated.
    highlight_labels = {"ratio10_seed42_step2000", "ratio10_seed42_step24500"}

    # Group records by (ratio, step) so that the 20-iter and 100-iter versions
    # of the same checkpoint are plotted as a pair, in training-phase order.
    records.sort(key=lambda r: (r["ratio"], r["step"], r["n_iter"]))

    fig, ax = plt.subplots(figsize=(11, 5))

    x_positions = []
    x_labels = []
    pos = 0
    rng = np.random.default_rng(0)  # for horizontal jitter only, does not affect the data itself

    for r in records:
        vals = np.array(r["lambda_max_samples"])
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        color = "#d62728" if r["label"] in highlight_labels else "#1f77b4"
        alpha = 0.9 if r["label"] in highlight_labels else 0.5

        ax.scatter(np.full_like(vals, pos) + jitter, vals, color=color, alpha=alpha, s=35, zorder=3)
        ax.hlines(vals.mean(), pos - 0.2, pos + 0.2, color="black", linewidth=1.5, zorder=4)

        cv = 100 * vals.std() / vals.mean()
        ax.text(pos, vals.max() + 0.05 * (vals.max() - vals.min() + 1), f"CV={cv:.0f}%",
                ha="center", fontsize=8)

        x_positions.append(pos)
        iter_tag = f"{r['n_iter']}it"
        x_labels.append(f"{int(r['ratio']*100)}%\nstep {r['step']}\n({iter_tag})")
        pos += 1

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel(r"$\lambda_{max}$ (10 repeated measurements, frozen weights)")
    ax.set_title("Repeated power-iteration measurements on fixed checkpoints:\n"
                  "resolved case (10% / step 2000) vs. persistent bimodal case (10% / step 24500)")
    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    fig.savefig(args.output.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved: {args.output} and {args.output.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
