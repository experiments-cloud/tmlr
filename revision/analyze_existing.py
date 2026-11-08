"""Re-derive every number in the manuscript that can be derived from the uploaded JSONs.
Usage: python3 -I analyze_existing.py <results_dir> <slq_dir> <out_json>
"""
import json, os, sys, glob, statistics as st
import numpy as np

res, slqdir, out = sys.argv[1], sys.argv[2], sys.argv[3]
SEEDS = [42, 1, 2, 3]; RATIOS = [50, 25, 10]
R = {}

def ms(x):
    return float(np.mean(x)), float(np.std(x, ddof=1))  # sample SD (ddof=1)

# ---- Table 2 + per-seed t_grok, t_mem ----
acc = {}
for r in RATIOS:
    acc[r] = {}
    for s in SEEDS:
        d = json.load(open(f"{res}/grokking_telemetry_ratio{r}_seed{s}_with_hessian.json"))
        steps, va, tl, vl = d["steps"], d["val_accuracy"], d["train_loss"], d["val_loss"]
        tg = next((st_ for st_, a in zip(steps, va) if a >= 0.9), None)
        # memorization proxy: first logged step where the (single-minibatch) training loss < 0.1
        tm = next((st_ for st_, l in zip(steps, tl) if l < 0.1), None)
        tm01 = next((st_ for st_, l in zip(steps, tl) if l < 0.01), None)
        # PI lambda trajectory
        cs, lam = d["checkpoint_steps"], d["lambda_max"]
        acc[r][s] = dict(final_acc=va[-1], max_acc=max(va), t_grok=tg, t_trainloss_lt_0p1=tm,
                         t_trainloss_lt_0p01=tm01, n_ckpt=len(cs),
                         pi_min=min(lam), pi_max=max(lam), pi_n_negative=sum(1 for x in lam if x < 0),
                         pi_at={str(k): lam[cs.index(k)] for k in (500, 2000, 24500) if k in cs},
                         trace_at={str(k): d["hessian_trace"][cs.index(k)] for k in (500, 2000, 24500) if k in cs},
                         last_step=steps[-1])
R["per_seed"] = acc
tab2 = {}
for r in RATIOS:
    fa = [acc[r][s]["final_acc"] for s in SEEDS]
    tg = [acc[r][s]["t_grok"] for s in SEEDS]
    tab2[r] = dict(final_acc_mean_sd=ms(fa), final_acc_sd_pop=float(np.std(fa)),
                   max_acc_over_training=max(acc[r][s]["max_acc"] for s in SEEDS),
                   t_grok=tg, t_grok_mean_sd=ms(tg) if None not in tg else None,
                   t_grok_sd_pop=float(np.std(tg)) if None not in tg else None)
R["table2"] = tab2

# ---- SLQ ----
slq = {}
for f in sorted(glob.glob(f"{slqdir}/checkpoints_ratio*_spectral_density.json")):
    d = json.load(open(f))
    ck = d["checkpoint"]; r = int(round(d["train_ratio"] * 100)); s = d["seed"]
    step = int(ck.replace(".pt", "").split("step_")[1])
    rv = np.array(d["ritz_values"]); w = np.array(d["weights"])
    nv, m = d["num_vectors"], d["lanczos_steps"]
    per_run_top = rv.reshape(nv, -1).max(1) if len(rv) == nv * m else None
    per_run_bot = rv.reshape(nv, -1).min(1) if len(rv) == nv * m else None
    N = 422 * 1000  # placeholder, replaced below if needed
    slq[(r, s, step)] = dict(
        lam_max=float(rv.max()), lam_min=float(rv.min()), loss=d["loss_at_checkpoint"],
        n_ritz=len(rv), per_run_top=None if per_run_top is None else per_run_top.tolist(),
        per_run_top_range=None if per_run_top is None else float(per_run_top.max() - per_run_top.min()),
        per_run_top_cv_pct=None if per_run_top is None else float(100 * per_run_top.std() / per_run_top.mean()),
        per_run_bottom=None if per_run_bot is None else per_run_bot.tolist(),
        weight_sum=float(w.sum()),
        weighted_mean=float(np.sum(rv * w)),
        mass_above_10=float(w[rv > 10].sum()), mass_abs_below_1=float(w[np.abs(rv) < 1].sum()),
        mass_negative=float(w[rv < 0].sum()))
R["slq"] = {f"r{r}_s{s}_t{t}": v for (r, s, t), v in slq.items()}
tab4 = {}
for r in RATIOS:
    tab4[r] = {}
    for t in (500, 2000, 24500):
        v = [slq[(r, s, t)]["lam_max"] for s in SEEDS if (r, s, t) in slq]
        tab4[r][t] = dict(values_by_seed={s: slq[(r, s, t)]["lam_max"] for s in SEEDS if (r, s, t) in slq},
                          mean_sd=ms(v), n=len(v))
R["table4"] = tab4
R["max_abs_neg_over_pos_ratio"] = max(abs(v["lam_min"]) / v["lam_max"] for v in slq.values())
R["worst_neg"] = min((v["lam_min"], k) for k, v in R["slq"].items())
R["slq_per_run_top_range_max"] = max((v["per_run_top_range"], k) for k, v in R["slq"].items() if v["per_run_top_range"] is not None)
R["slq_per_run_top_cv_max"] = max((v["per_run_top_cv_pct"], k) for k, v in R["slq"].items() if v["per_run_top_cv_pct"] is not None)

# ---- noise diagnostic (Table 3) ----
nd = {}
for f in sorted(glob.glob(f"{res}/*noise_diagnostic.json")):
    d = json.load(open(f)); l = d["lambda_max_samples"]
    nd[os.path.basename(f)] = dict(n_iter=d.get("num_iterations", 20), fixed_batch=d.get("fixed_batch", "field absent (=fresh batch)"),
                                  cv_pop_pct=100 * st.pstdev(l) / st.mean(l), mean=st.mean(l), min=min(l), max=max(l),
                                  samples=l)
R["noise_diag"] = nd

# ---- ablation ----
ab = json.load(open(f"{res}/optimizer_ablation_results.json"))
R["ablation_file"] = {k: dict(n_points=len(v["step"]), keys=list(v.keys()), final_acc=v["val_acc"][-1],
                             peak_lam=max(v["lambda_max"]), min_lam=min(v["lambda_max"])) for k, v in ab.items()}

json.dump(R, open(out, "w"), indent=1, default=str)

# --- print summary ---
print("TABLE 2"); [print(r, tab2[r]) for r in RATIOS]
print("\nper-seed t_grok / train-loss<0.1 / <0.01 / PI negatives / PI min")
for r in RATIOS:
    for s in SEEDS:
        a = acc[r][s]; print(r, s, a["t_grok"], a["t_trainloss_lt_0p1"], a["t_trainloss_lt_0p01"], a["pi_n_negative"], round(a["pi_min"], 2), "final", round(a["final_acc"], 4), "max", round(a["max_acc"], 4), "n_ckpt", a["n_ckpt"])
print("\nTABLE 4 (SLQ max Ritz)")
for r in RATIOS:
    for t in (500, 2000, 24500):
        x = tab4[r][t]; print(r, t, {k: round(v, 1) for k, v in x["values_by_seed"].items()}, [round(v, 1) for v in x["mean_sd"]], x["n"])
print("\nSLQ per-checkpoint: top, bottom, per-run top range, PI at same step (different batch)")
for r in RATIOS:
    for s in SEEDS:
        for t in (500, 2000, 24500):
            v = slq[(r, s, t)]
            print(f"r{r} s{s} t{t}: slq_top={v['lam_max']:.2f} slq_min={v['lam_min']:.2f} run_top_range={v['per_run_top_range']:.3f} cv%={v['per_run_top_cv_pct']:.3f} PI={acc[r][s]['pi_at'].get(str(t))!s:.8} loss={v['loss']:.4f} mass>10={v['mass_above_10']:.2e}")
print("\nmax |lam_min|/lam_max:", R["max_abs_neg_over_pos_ratio"], "worst neg:", R["worst_neg"])
print("max per-run top range:", R["slq_per_run_top_range_max"], "max cv:", R["slq_per_run_top_cv_max"])
print("\nablation:", R["ablation_file"])
