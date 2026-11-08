"""New figures for the revision, built only from the uploaded JSONs.
Usage: python3 -I make_new_figures.py <results_dir> <slq_dir> <out_dir>
  figure5.png : 25% density, per seed: val accuracy, single-minibatch train loss, and curvature
                (PI trajectory on the run's fixed evaluation batch; SLQ top Ritz value at 500/2000/24500)
  figure6.png : SLQ spectral density (Gaussian-broadened pooled Ritz values, sigma=1.0), seed 42, 9 checkpoints
"""
import json, sys, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res, slqdir, outd = sys.argv[1:4]
SEEDS = [42, 1, 2, 3]
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4})

def slq_top(r, s, t):
    f = f"{slqdir}/checkpoints_ratio{r}_seed{s}_model_step_{t}_spectral_density.json"
    return max(json.load(open(f))["ritz_values"])

# ---------- Figure 5 ----------
fig, ax = plt.subplots(3, 4, figsize=(11, 6.4), sharex=True, sharey="row")
for j, s in enumerate(SEEDS):
    d = json.load(open(f"{res}/grokking_telemetry_ratio25_seed{s}_with_hessian.json"))
    st = np.array(d["steps"]); va = np.array(d["val_accuracy"]); tl = np.array(d["train_loss"])
    cs = np.array(d["checkpoint_steps"]); lam = np.array(d["lambda_max"])
    tg = st[np.argmax(va >= 0.9)]
    ax[0, j].plot(st, va, color=C[j]); ax[0, j].set_title(f"seed {s}  ($t_{{\\mathrm{{grok}}}}$ = {tg:,})", color=INK)
    ax[1, j].semilogy(st, tl, color=C[j])
    ax[2, j].plot(cs, lam, color=C[j], marker="o", ms=2.2, lw=1.0, label="PI, 20 it. (run's eval. batch)")
    sl = [slq_top(25, s, t) for t in (500, 2000)]
    ax[2, j].scatter([500, 2000], sl, s=46, facecolor="white", edgecolor=INK, lw=1.3, zorder=5,
                     label="SLQ top Ritz value (own batch)")
    for a in ax[:, j]:
        a.axvline(tg, color=INK2, ls="--", lw=0.9)
    ax[2, j].text(0.98, 0.03, f"step 24,500:\nSLQ {slq_top(25, s, 24500):.0f}, PI {lam[list(cs).index(24500)]:.0f}",
                  transform=ax[2, j].transAxes, ha="right", va="bottom", fontsize=7, color=INK2)
ax[0, 0].set_ylabel("validation accuracy"); ax[1, 0].set_ylabel("train loss\n(one mini-batch, logged)")
ax[2, 0].set_ylabel(r"$\lambda_{\max}$ estimate")
for a in ax[2]: a.set_xlabel("optimization step")
ax[2, 0].set_xlim(0, 5000)
ax[2, 0].legend(loc="upper left", fontsize=6.8, frameon=False)
fig.tight_layout()
fig.savefig(f"{outd}/figure5.png", dpi=250)

# ---------- Figure 6 ----------
fig, ax = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
grid = np.linspace(-20, 160, 1801); sig = 1.0
for i, r in enumerate([50, 25, 10]):
    for k, t in enumerate([500, 2000, 24500]):
        d = json.load(open(f"{slqdir}/checkpoints_ratio{r}_seed42_model_step_{t}_spectral_density.json"))
        rv = np.array(d["ritz_values"]); w = np.array(d["weights"])
        dens = (w[:, None] * np.exp(-(grid[None, :] - rv[:, None]) ** 2 / (2 * sig ** 2))).sum(0) / (sig * np.sqrt(2 * np.pi))
        ax[i].semilogy(grid, np.maximum(dens, 1e-9), color=C[k], label=f"step {t:,}")
    ax[i].set_title(f"{r}% data, seed 42", color=INK); ax[i].set_xlabel("eigenvalue")
    ax[i].set_ylim(1e-7, 2)
ax[0].set_ylabel("density (Gaussian-broadened, $\\sigma$=1)")
ax[0].legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(f"{outd}/figure6.png", dpi=250)
print("ok")
