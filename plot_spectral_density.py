"""
plot_spectral_density.py

Reads one or more *_spectral_density.json files (produced by spectral_density_slq.py)
and renders the weighted eigenvalue density (Gaussian-kernel smoothed, weighted by the
SLQ quadrature weights) for each checkpoint, arranged in a grid (rows = data ratio,
columns = training phase) for direct visual comparison.

Usage:
    python plot_spectral_density.py \
        --files ratio50_step500.json ratio50_step2000.json ratio50_step24500.json \
                ratio25_step500.json ratio25_step2000.json ratio25_step24500.json \
                ratio10_step500.json ratio10_step2000.json ratio10_step24500.json \
        --output spectral_density_grid.png
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def load_density(path):
    with open(path) as f:
        d = json.load(f)
    return np.array(d["ritz_values"]), np.array(d["weights"]), d


def plot_single(ax, ritz_values, weights, title):
    # Weighted Gaussian KDE over the pooled Ritz values (approximates the eigenvalue density)
    if len(ritz_values) < 2 or np.allclose(ritz_values, ritz_values[0]):
        ax.axvline(ritz_values[0] if len(ritz_values) else 0, color="crimson")
        ax.set_title(title + " (degenerada)")
        return

    # Guard against all-zero/negative weights collapsing the KDE
    w = np.clip(weights, 1e-12, None)
    kde = gaussian_kde(ritz_values, weights=w, bw_method=0.15)

    lo, hi = ritz_values.min(), ritz_values.max()
    pad = 0.1 * (hi - lo + 1e-6)
    grid = np.linspace(lo - pad, hi + pad, 400)
    density = kde(grid)

    ax.fill_between(grid, density, alpha=0.3, color="#1f77b4")
    ax.plot(grid, density, color="#1f77b4", linewidth=1.5)

    # Mark the top eigenvalue candidates (largest Ritz values) as ticks
    top_eigs = np.sort(ritz_values)[::-1][:5]
    for te in top_eigs:
        ax.axvline(te, color="crimson", alpha=0.4, linewidth=0.8, linestyle="--")

    ax.set_title(title, fontsize=10)
    ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True,
                         help="List of *_spectral_density.json files, in row-major grid order.")
    parser.add_argument("--ncols", type=int, default=3)
    parser.add_argument("--output", type=str, default="spectral_density_grid.png")
    args = parser.parse_args()

    n = len(args.files)
    ncols = args.ncols
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, path in enumerate(args.files):
        ritz_values, weights, meta = load_density(path)
        label = f"ratio={int(meta['train_ratio']*100)}% seed={meta['seed']}\ncheckpoint: {meta['checkpoint'].split('/')[-1]}"
        plot_single(axes[i], ritz_values, weights, label)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Estimación de densidad espectral del Hessiano (SLQ) — líneas discontinuas: top-5 Ritz values", fontsize=11)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Figura guardada en: {args.output}")


if __name__ == "__main__":
    main()
