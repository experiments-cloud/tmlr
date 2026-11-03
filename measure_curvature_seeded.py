"""
measure_curvature_seeded.py  (NEW, revision experiment; Appendix B)

For every selected checkpoint of one training run, measures the top Hessian eigenvalue
with a protocol that separates batch variability from changes in the weights:

  * objective: mean CE at the final position, model.eval(), no weight-decay term
    (identical to hessian_topology.py / spectral_density_slq.py);
  * K mini-batches of 512 training examples drawn ONCE with a seeded generator and
    REUSED at every checkpoint (so the batch is not a source of change along the trajectory);
    on each batch: Lanczos (full reorthogonalization, m steps) -> top Ritz value + residual;
  * the FULL training split (mean loss over all training examples): Lanczos (m_full steps)
    -> two largest Ritz values + residuals (batch-independent reference, near-degeneracy check).

Resumable: re-running skips checkpoints already present in the output JSON.

Usage:
  python3 measure_curvature_seeded.py --repo . --run_name ext_ratio20_seed42 --train_ratio 0.2 --seed 42
"""
import argparse, glob, json, os, sys, time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--train_ratio", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--m", type=int, default=25)
    ap.add_argument("--m_full", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--batch_seed", type=int, default=12345)
    ap.add_argument("--dense_until", type=int, default=8000, help="measure every --dense_stride steps up to here")
    ap.add_argument("--dense_stride", type=int, default=200)
    ap.add_argument("--sparse_stride", type=int, default=1000, help="then every --sparse_stride steps, plus the last checkpoint")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sys.path.insert(0, a.repo)
    import torch, torch.nn as nn
    from generate_dataset import ModularArithmeticDataset
    from model_architecture import ToyTransformer
    from hessian_topology import compute_hvp
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        _sdp = lambda: sdpa_kernel(SDPBackend.MATH)
    except ImportError:
        _sdp = lambda: torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)

    out_path = a.out or f"curvature_{a.run_name}.json"
    res = json.load(open(out_path)) if os.path.exists(out_path) else {"args": vars(a), "checkpoints": {}}

    ds = ModularArithmeticDataset(p=97, split="train", train_ratio=a.train_ratio, seed=a.seed)
    X = torch.stack([ds[i][0] for i in range(len(ds))]); Y = torch.stack([ds[i][1] for i in range(len(ds))])
    gen = torch.Generator().manual_seed(a.batch_seed)
    batches = [torch.randperm(len(ds), generator=gen)[:a.batch_size] for _ in range(a.K)]
    res["batch_heads"] = [b[:8].tolist() for b in batches]
    res["torch_version"] = torch.__version__; res["threads"] = torch.get_num_threads()
    torch.manual_seed(a.batch_seed)  # Lanczos start vectors

    model = ToyTransformer(vocab_size=ds.vocab_size); model.eval()
    crit = nn.CrossEntropyLoss()
    N = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def hvp_fn(idx):
        chunks = [idx[i:i + a.batch_size] for i in range(0, len(idx), a.batch_size)]
        losses = []
        for c in chunks:
            with _sdp():
                lg = model(X[c])
                losses.append((crit(lg[:, -1, :], Y[c][:, -1]), len(c) / len(idx)))
        def f(v):
            out = torch.zeros_like(v)
            for l, w in losses:
                out += w * compute_hvp(model, l, v)
            return out
        return f, float(sum(l.item() * w for l, w in losses))

    def lanczos(H, m):
        v = torch.randn(N); v = v / torch.norm(v); vp = torch.zeros(N); bp = 0.0
        V = [v.clone()]; al, be = [], []; beta = 0.0
        for j in range(m):
            w = H(v); alpha = torch.dot(w, v).item(); al.append(alpha)
            w = w - alpha * v - bp * vp
            for vi in V:
                w = w - torch.dot(w, vi) * vi
            beta = torch.norm(w).item()
            if j == m - 1 or beta < 1e-10:
                break
            be.append(beta); vp = v; v = w / beta; V.append(v.clone()); bp = beta
        T = np.diag(al) + np.diag(be, 1) + np.diag(be, -1)
        th, S = np.linalg.eigh(T); o = np.argsort(th)[::-1][:2]
        return th[o].tolist(), (beta * np.abs(S[-1, o])).tolist()

    files = glob.glob(os.path.join("checkpoints", a.run_name, "model_step_*.pt"))
    steps_all = sorted(int(f.split("model_step_")[1].split(".pt")[0]) for f in files)
    if not steps_all:
        print("No checkpoints found in", os.path.join("checkpoints", a.run_name)); return
    sel = [s for s in steps_all if (s <= a.dense_until and s % a.dense_stride == 0) or s % a.sparse_stride == 0 or s == steps_all[-1]]
    print(f"{a.run_name}: {len(sel)} checkpoints selected of {len(steps_all)} | torch {torch.__version__} threads {torch.get_num_threads()}", flush=True)

    full_idx = torch.arange(len(ds)); t0 = time.time()
    for s in sel:
        if str(s) in res["checkpoints"]:
            continue
        model.load_state_dict(torch.load(os.path.join("checkpoints", a.run_name, f"model_step_{s}.pt"), map_location="cpu"))
        rec = {"batch_top": [], "batch_resid": [], "batch_loss": []}
        for b in batches:
            H, lv = hvp_fn(b); th, r = lanczos(H, a.m)
            rec["batch_top"].append(th[0]); rec["batch_resid"].append(r[0]); rec["batch_loss"].append(lv)
        H, lv = hvp_fn(full_idx); th, r = lanczos(H, a.m_full)
        rec.update(full_top2=th, full_resid2=r, full_loss=lv,
                   batch_mean=float(np.mean(rec["batch_top"])), batch_sd=float(np.std(rec["batch_top"])))
        res["checkpoints"][str(s)] = rec
        json.dump(res, open(out_path, "w"), indent=1)
        print(f"[{time.time()-t0:7.0f}s] step {s}: batches {np.round(rec['batch_top'],1)} | full {np.round(th,2)} resid {np.format_float_scientific(max(r),2)}", flush=True)
    print("done:", out_path)


if __name__ == "__main__":
    main()
