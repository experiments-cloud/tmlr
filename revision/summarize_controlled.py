"""Summarize controlled_pi_vs_slq.py outputs into the numbers used in the paper.
Usage: python3 -I summarize_controlled.py out.json ctrl1.json [ctrl2.json ...]
CV uses the sample SD (ddof=1), as everywhere in the paper.
"""
import json, sys
import numpy as np

def cv(x):
    x = np.asarray(x, float); return float(100 * x.std(ddof=1) / x.mean())

rows = {}
for f in sys.argv[2:]:
    d = json.load(open(f)); a = d["args"]
    key = f"r{int(round(a['train_ratio']*100))}_s{a['seed']}_{a['checkpoint'].split('model_step_')[1].replace('.pt','')}"
    B = d["batches"]
    pi20 = np.array([b["pi20"] for b in B]); pi100 = np.array([b["pi100"] for b in B]); slq = np.array([b["slq_top_per_run"] for b in B])
    r = dict(
        K=len(B), loss_per_batch=[b["loss"] for b in B],
        # within-batch (fixed Hessian) spread
        within_range_pi20=float((pi20.max(1) - pi20.min(1)).max()),
        within_range_pi100=float((pi100.max(1) - pi100.min(1)).max()),
        within_range_slq=float((slq.max(1) - slq.min(1)).max()),
        # same-batch agreement: SLQ top (max over runs) minus PI100 (max over inits)
        max_abs_diff_slq_vs_pi100=float(np.abs(slq.max(1) - pi100.max(1)).max()),
        max_abs_diff_slq_vs_pi20=float(np.abs(slq.max(1) - pi20.max(1)).max()),
        max_rel_diff_slq_vs_pi20=float((np.abs(slq.max(1) - pi20.max(1)) / slq.max(1)).max()),
        # across-batch spread (one estimate per batch: first init / first run, mimicking a single measurement)
        per_batch_slq=slq.max(1).tolist(), per_batch_pi20_first=pi20[:, 0].tolist(), per_batch_pi100_first=pi100[:, 0].tolist(),
        cv_across_batches_slq=cv(slq.max(1)), cv_across_batches_pi20=cv(pi20[:, 0]), cv_across_batches_pi100=cv(pi100[:, 0]),
        min_across_batches_slq=float(slq.max(1).min()), max_across_batches_slq=float(slq.max(1).max()),
        lanczos_long_batch0=B[0].get("lanczos_long"),
    )
    if "full_train" in d:
        r["full_train"] = d["full_train"]
    rows[key] = r
    ll = B[0]["lanczos_long"]; m = max(int(k) for k in ll)
    print(f"\n== {key}  K={r['K']}")
    print(f" within-batch max range: PI20 {r['within_range_pi20']:.3f} PI100 {r['within_range_pi100']:.3f} SLQ {r['within_range_slq']:.3f}")
    print(f" same-batch |SLQ-PI100| max {r['max_abs_diff_slq_vs_pi100']:.3f}; |SLQ-PI20| max {r['max_abs_diff_slq_vs_pi20']:.3f} (rel {100*r['max_rel_diff_slq_vs_pi20']:.2f}%)")
    print(f" across batches: SLQ {np.round(r['per_batch_slq'],1)}  CV {r['cv_across_batches_slq']:.1f}%")
    print(f"                 PI20 {np.round(r['per_batch_pi20_first'],1)}  CV {r['cv_across_batches_pi20']:.1f}%")
    print(f"                 PI100 CV {r['cv_across_batches_pi100']:.1f}%")
    for k in sorted(ll, key=int):
        print(f" batch0 Lanczos m={k}: top5 {np.round(ll[k]['top5'],3)} resid {np.round(ll[k]['resid'],4)}")
    if "full_train" in d:
        fl = d["full_train"]["lanczos_long"]; mk = max(fl, key=int)
        print(f" FULL split m={mk}: top5 {np.round(fl[mk]['top5'],3)} resid {np.round(fl[mk]['resid'],4)} PI100 {np.round(d['full_train']['pi100'],3)}")
json.dump(rows, open(sys.argv[1], "w"), indent=1)
