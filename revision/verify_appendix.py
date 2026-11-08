"""Verify Table 3 col.3, Table 7 (fine SLQ steps), Table 8 / Appendix A from the second upload.
Usage: python3 -I verify_appendix.py <apx_dir> <slq36_dir>"""
import json, sys, glob, os
import numpy as np
apx, slq36 = sys.argv[1:3]
SEEDS = [42, 1, 2, 3]
print("== Table 3, fixed batch")
for f in sorted(glob.glob(f"{apx}/*fixedbatch*")):
    d = json.load(open(f)); l = np.array(d["lambda_max_samples"])
    print(os.path.basename(f)[:45], d.get("fixed_batch"), d.get("num_iterations"), "CV_pop=%.3f" % (100 * l.std() / l.mean()), np.round(np.sort(l), 2))
print("\n== Table 7 (SLQ max Ritz, 25%)")
def top(s, t):
    for dd in (apx, slq36):
        f = f"{dd}/checkpoints_ratio25_seed{s}_model_step_{t}_spectral_density.json"
        if os.path.exists(f):
            d = json.load(open(f)); rv = np.array(d["ritz_values"])
            return rv.max(), (np.ptp(rv.reshape(d["num_vectors"], -1).max(1)) if len(rv) == d["num_vectors"] * d["lanczos_steps"] else None)
    return None, None
for t in (500, 1000, 1500, 2000, 2500, 3000, 3500):
    v = [top(s, t) for s in SEEDS]; x = np.array([a for a, _ in v])
    print(t, np.round(x, 1), "mean %.1f popSD %.1f sampSD %.1f" % (x.mean(), x.std(), x.std(ddof=1)), "max per-run range %.4f" % max(b for _, b in v))
print("\n== Appendix A")
R = {s: json.load(open(f"{apx}/optimizer_ablation_results_seed{s}.json")) for s in SEEDS}
for opt in ("AdamW", "Adam", "SGD"):
    fa = np.array([R[s][opt]["val_acc"][-1] for s in SEEDS])
    pk = np.array([max(R[s][opt]["lambda_max"]) for s in SEEDS])
    mn = np.array([min(R[s][opt]["lambda_max"]) for s in SEEDS])
    wn = [np.array(R[s][opt]["weight_l2_norm"]) for s in SEEDS]
    trough = np.array([w.min() for w in wn]); final = np.array([w[-1] for w in wn]); init = np.array([w[0] for w in wn])
    st = [R[s][opt]["step"] for s in SEEDS]
    trough_step = [st[i][int(np.argmin(wn[i]))] for i in range(4)]
    print(opt, "steps", st[0][0], st[0][-1], len(st[0]))
    print("  final acc", np.round(fa, 4), "mean %.4f pop %.4f samp %.4f" % (fa.mean(), fa.std(), fa.std(ddof=1)))
    print("  peak lam", np.round(pk, 1), "mean %.1f pop %.1f samp %.1f" % (pk.mean(), pk.std(), pk.std(ddof=1)), "min lam", np.round(mn, 3))
    print("  w-norm init", np.round(init, 2), "trough", np.round(trough, 4), "at", trough_step, "mean %.4f pop %.4f" % (trough.mean(), trough.std()))
    print("  w-norm final", np.round(final, 4), "mean %.4f pop %.4f samp %.4f" % (final.mean(), final.std(), final.std(ddof=1)))
    if opt == "AdamW":
        late = [w[np.array(st[i]) >= 3000] for i, w in enumerate(wn)]
        print("  AdamW norm range after step 3000:", round(min(l.min() for l in late), 2), round(max(l.max() for l in late), 2))
    else:
        # step at which norm first < initial/1000
        print("  first step norm < init/1000:", [st[i][int(np.argmax(wn[i] < init[i] / 1000))] if (wn[i] < init[i] / 1000).any() else None for i in range(4)])
