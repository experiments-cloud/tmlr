"""
controlled_pi_vs_slq.py  (NEW, written for the TMLR revision; not part of the submitted results)

Controlled comparison of power iteration (PI) and Lanczos/SLQ on EXACTLY the same
empirical Hessians, at one frozen checkpoint.

Objective measured (identical to hessian_topology.py / spectral_density_slq.py of the repo):
  * loss  = mean cross-entropy of the FINAL position only: CE(logits[:, -1, :], y[:, -1])
  * model.eval()  (dropout disabled), unfused (MATH) attention for the forward pass
  * no weight-decay/L2 term in the loss whose Hessian is measured
  * batches of 512 training examples (BATCH_SIZE in hessian_topology.py)

Design, per checkpoint:
  For each of K mini-batches b_1..b_K (drawn with a seeded generator, without replacement
  inside a batch):
     - PI, 20 iterations, from P20 independent random v0     (repo protocol)
     - PI, 100 iterations, from P100 independent random v0
     - SLQ: R independent Lanczos runs of m steps (full reorth.) -> top Ritz value per run
  Plus, on batch b_1 only (fixed-Hessian convergence / near-degeneracy check):
     - one Lanczos run of M_LONG steps; top-5 Ritz values with residual norms
       ||H y - theta y|| = beta_m |e_m^T s| after m = 10, 20, 30, 50, M_LONG steps
  Plus a batch-independent reference: the same Lanczos run on the FULL training split
  (all training examples, chunked; the loss is the mean over all of them).

Usage:
  python3 controlled_pi_vs_slq.py --repo /path/to/repo --checkpoint ckpt.pt \
      --train_ratio 0.1 --seed 42 --out out.json [--K 10 --P20 5 --P100 3 --R 10 --m 50 --M_LONG 100]
"""
import argparse, json, sys, time, math
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train_ratio", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--P20", type=int, default=5)
    ap.add_argument("--P100", type=int, default=3)
    ap.add_argument("--R", type=int, default=10)
    ap.add_argument("--m", type=int, default=50)
    ap.add_argument("--M_LONG", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--rng_seed", type=int, default=12345)
    ap.add_argument("--skip_full", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, a.repo)
    import torch, torch.nn as nn
    from generate_dataset import ModularArithmeticDataset
    from model_architecture import ToyTransformer
    from hessian_topology import compute_hvp
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        _sdp = lambda: sdpa_kernel(SDPBackend.MATH)
    except ImportError:  # PyTorch < 2.3, same fallback as hessian_topology.py
        _sdp = lambda: torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)

    torch.manual_seed(a.rng_seed)
    gen = torch.Generator().manual_seed(a.rng_seed)
    dev = torch.device("cpu")  # CPU on purpose: same device as the original runs
    ds = ModularArithmeticDataset(p=97, split="train", train_ratio=a.train_ratio, seed=a.seed)
    X = torch.stack([ds[i][0] for i in range(len(ds))]); Y = torch.stack([ds[i][1] for i in range(len(ds))])
    model = ToyTransformer(vocab_size=ds.vocab_size).to(dev)
    model.load_state_dict(torch.load(a.checkpoint, map_location=dev))
    model.eval()
    crit = nn.CrossEntropyLoss()
    N = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def make_loss(idx):
        with _sdp():
            lg = model(X[idx])
            return crit(lg[:, -1, :], Y[idx][:, -1])

    def hvp_fn(idx):
        """Returns a function v -> H v for the mean-CE loss over examples idx.
        For the full split the loss is evaluated in chunks and re-weighted to the global mean."""
        if len(idx) <= a.batch_size:
            loss = make_loss(idx)
            return (lambda v: compute_hvp(model, loss, v)), loss.item()
        chunks = [idx[i:i + a.batch_size] for i in range(0, len(idx), a.batch_size)]
        losses = [(make_loss(c), len(c) / len(idx)) for c in chunks]
        def f(v):
            out = torch.zeros_like(v)
            for l, w in losses:
                out += w * compute_hvp(model, l, v)
            return out
        return f, float(sum(l.item() * w for l, w in losses))

    def power_iteration(H, iters):  # identical update / Rayleigh quotient to hessian_topology.py
        v = torch.randn(N); v = v / torch.norm(v); lam = 0.0
        for _ in range(iters):
            Hv = H(v); lam = torch.dot(v, Hv).item(); v = Hv / (torch.norm(Hv) + 1e-8)
        return lam

    def lanczos(H, m, checkpoints=()):
        """Lanczos with full reorthogonalization (same recurrence as spectral_density_slq.py).
        Returns top Ritz value and, for each k in checkpoints, the top-5 Ritz values and residuals."""
        v = torch.randn(N); v = v / torch.norm(v); vp = torch.zeros(N); bp = 0.0
        V = [v.clone()]; al, be = [], []; snaps = {}
        for j in range(m):
            w = H(v); alpha = torch.dot(w, v).item(); al.append(alpha)
            w = w - alpha * v - bp * vp
            for vi in V:
                w = w - torch.dot(w, vi) * vi
            beta = torch.norm(w).item()
            k = j + 1
            if k in checkpoints or k == m:
                T = np.diag(al) + np.diag(be, 1) + np.diag(be, -1)
                th, S = np.linalg.eigh(T)
                order = np.argsort(th)[::-1][:5]
                snaps[k] = dict(top5=th[order].tolist(), resid=(beta * np.abs(S[-1, order])).tolist(),
                                weight_top5=(S[0, order] ** 2).tolist())
            if j == m - 1 or beta < 1e-10:
                break
            be.append(beta); vp = v; v = w / beta; V.append(v.clone()); bp = beta
        return snaps

    out = dict(args=vars(a), N=N, n_train=len(ds), objective="mean CE over final position; model.eval(); no WD term",
               batches=[])
    print(f"torch {torch.__version__} | threads {torch.get_num_threads()} | checkpoint {a.checkpoint}", flush=True)
    out["torch_version"] = torch.__version__; out["threads"] = torch.get_num_threads()
    t0 = time.time()
    perm_batches = [torch.randperm(len(ds), generator=gen)[:a.batch_size] for _ in range(a.K)]
    for bi, idx in enumerate(perm_batches):
        H, lv = hvp_fn(idx)
        rec = dict(batch=bi, loss=lv, idx_head=idx[:8].tolist())
        rec["pi20"] = [power_iteration(H, 20) for _ in range(a.P20)]
        rec["pi100"] = [power_iteration(H, 100) for _ in range(a.P100)]
        rec["slq_top_per_run"] = [lanczos(H, a.m)[a.m]["top5"][0] for _ in range(a.R)]
        if bi == 0:
            rec["lanczos_long"] = lanczos(H, a.M_LONG, checkpoints=(10, 20, 30, 50, a.M_LONG))
        out["batches"].append(rec)
        print(f"[{time.time()-t0:7.1f}s] batch {bi}: loss={lv:.4f} PI20={np.round(rec['pi20'],1)} "
              f"PI100={np.round(rec['pi100'],1)} SLQtop={np.round(rec['slq_top_per_run'],2)}", flush=True)
        json.dump(out, open(a.out, "w"), indent=1)
    if not a.skip_full:
        idx = torch.arange(len(ds))
        H, lv = hvp_fn(idx)
        out["full_train"] = dict(loss=lv, lanczos_long=lanczos(H, a.M_LONG, checkpoints=(20, 50, a.M_LONG)),
                                 pi100=[power_iteration(H, 100) for _ in range(2)])
        print(f"[{time.time()-t0:7.1f}s] full split: {out['full_train']['lanczos_long'][a.M_LONG]['top5']} PI100={out['full_train']['pi100']}", flush=True)
    out["elapsed_s"] = time.time() - t0
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
