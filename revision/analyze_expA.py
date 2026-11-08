"""Appendix B analysis: new densities 20% and 35%, 4 seeds.
Usage: python3 -I analyze_expA.py <expA_dir> <out_json> <fig_png>
If a curvature JSON is missing, values are parsed from logs/curv_<run>.log (rounded: batches 0.1, full 0.01),
and the run is flagged with source='log'.
"""
import json, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D, OUT, FIG = sys.argv[1:4]
SEEDS = [42, 1, 2, 3]; RATIOS = [20, 35]

def load_curv(run):
    f = f"{D}/curvature_{run}.json"
    if os.path.exists(f):
        c = json.load(open(f))["checkpoints"]
        rows = {int(k): dict(b=v["batch_top"], full=v["full_top2"], res=max(v["full_resid2"])) for k, v in c.items()}
        return rows, "json"
    rows = {}
    for line in open(f"{D}/logs/curv_{run}.log", errors="ignore"):
        m = re.search(r"step (\d+): batches \[([^\]]*)\] \| full \[([^\]]*)\] resid (\S+)", line)
        if m:
            rows[int(m.group(1))] = dict(b=[float(x) for x in m.group(2).split()], full=[float(x) for x in m.group(3).split()], res=float(m.group(4)))
    return rows, "log"

R = {}
for r in RATIOS:
    for s in SEEDS:
        run = f"ext_ratio{r}_seed{s}"
        t = json.load(open(f"{D}/grokking_telemetry_{run}.json"))
        st = np.array(t["steps"]); va = np.array(t["val_accuracy"]); ta = np.array(t["train_full_accuracy"])
        first = lambda a, th: int(st[np.argmax(a >= th)]) if (a >= th).any() else None
        rows, src = load_curv(run)
        ks = np.array(sorted(rows)); full1 = np.array([rows[k]["full"][0] for k in ks]); full2 = np.array([rows[k]["full"][1] for k in ks])
        bm = np.array([np.mean(rows[k]["b"]) for k in ks]); bsd = np.array([np.std(rows[k]["b"], ddof=1) for k in ks])
        tg, tm = first(va, 0.9), first(ta, 0.999)
        dense = ks <= 8000
        ipk = int(np.argmax(np.where(dense, full1, -np.inf)))
        early = (ks >= 400) & (ks <= 1600)
        imin = int(np.argmin(np.where(early, full1, np.inf)))
        jump = np.abs(np.diff(full1[dense])) / np.minimum(full1[dense][:-1], full1[dense][1:])
        rel_prev = np.abs(np.diff(full1[dense])) / full1[dense][:-1]  # |l(t+200)-l(t)|/l(t), steps <= 8000
        ratio_cons = np.maximum(full1[dense][1:], full1[dense][:-1]) / np.minimum(full1[dense][1:], full1[dense][:-1])
        R[run] = dict(source=src, density=r, seed=s, t_mem_train999=tm, t_train99=first(ta, 0.99), t_val01=first(va, 0.1), t_grok=tg,
                      delay=(tg - tm) if (tg is not None and tm is not None) else None,
                      final_val=float(va[-1]), final_train=float(ta[-1]),
                      full_l1_at0=float(full1[0]), full_l1_min_400_1600=float(full1[imin]), step_min=int(ks[imin]),
                      full_l1_peak_le8000=float(full1[ipk]), step_peak=int(ks[ipk]), peak_minus_tgrok=int(ks[ipk]) - tg,
                      full_l1_near_tgrok=float(full1[np.argmin(abs(ks - tg))]), step_near_tgrok=int(ks[np.argmin(abs(ks - tg))]),
                      full_l1_last=float(full1[-1]),
                      max_rel_jump_200steps=float(jump.max()), median_rel_jump=float(np.median(jump)),
                      median_rel_change_prev_le8000=float(np.median(rel_prev)), max_ratio_consecutive_le8000=float(ratio_cons.max()),
                      ratio_l2_l1_median=float(np.median(full2 / full1)), frac_ratio_gt_095=float(np.mean(full2 / full1 > 0.95)),
                      n_ckpt=int(len(ks)), max_full_resid=float(max(rows[k]["res"] for k in ks)),
                      batch_cv_median=float(np.median(100 * bsd / bm)), batch_cv_max=float(np.max(100 * bsd / bm)),
                      batchmean_over_full_median=float(np.median(bm / full1)),
                      traj=dict(steps=ks.tolist(), full1=full1.tolist(), full2=full2.tolist(), batch_mean=bm.tolist(), batch_min=[min(rows[k]["b"]) for k in ks], batch_max=[max(rows[k]["b"]) for k in ks]),
                      acc=dict(steps=st.tolist(), val=va.tolist(), train=ta.tolist()))
json.dump(R, open(OUT, "w"), indent=1)

# ---- print summary ----
keys = ["source", "t_mem_train999", "t_grok", "delay", "full_l1_at0", "full_l1_min_400_1600", "step_min", "full_l1_peak_le8000", "step_peak", "peak_minus_tgrok",
        "full_l1_near_tgrok", "full_l1_last", "max_rel_jump_200steps", "ratio_l2_l1_median", "frac_ratio_gt_095", "batch_cv_median", "batchmean_over_full_median", "max_full_resid", "final_val", "final_train"]
for run, v in R.items():
    print(run, {k: (round(v[k], 3) if isinstance(v[k], float) else v[k]) for k in keys})

# ---- figure ----
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]; INK2 = "#52514e"
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.color": "#e6e5e0", "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(4, 4, figsize=(11, 8.2), sharex="row")
for i, r in enumerate(RATIOS):
    for j, s in enumerate(SEEDS):
        v = R[f"ext_ratio{r}_seed{s}"]; a0 = ax[2 * i, j]; a1 = ax[2 * i + 1, j]
        xmax = 8000
        a0.plot(v["acc"]["steps"], v["acc"]["train"], color=INK2, lw=1.2, ls=":", label="train (full set)")
        a0.plot(v["acc"]["steps"], v["acc"]["val"], color=C[j], lw=1.4, label="validation")
        t = v["traj"]; k = np.array(t["steps"])
        a1.fill_between(k, t["batch_min"], t["batch_max"], color=C[j], alpha=0.18, lw=0, label="4 fixed batches (range)")
        a1.plot(k, t["full1"], color=C[j], lw=1.3, marker="o", ms=2, label=r"full training set, $\lambda_1$")
        for a in (a0, a1):
            a.axvline(v["t_grok"], color=INK2, ls="--", lw=0.9)
            if v["t_mem_train999"] is not None:
                a.axvline(v["t_mem_train999"], color=INK2, ls=":", lw=0.9)
            a.set_xlim(0, xmax)
        a0.set_title(f"{r}% data, seed {s}")
        if j == 0:
            a0.set_ylabel("accuracy"); a1.set_ylabel(r"top eigenvalue")
        if i == 1:
            a1.set_xlabel("optimization step")
ax[0, 0].legend(fontsize=6.5, frameon=False, loc="center right"); ax[1, 0].legend(fontsize=6.5, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(FIG, dpi=230)
print("figure ->", FIG)
