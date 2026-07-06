"""
spectral_density_slq.py

Estimates the FULL eigenvalue density (spectral density) of the empirical
Hessian for a given checkpoint, using Stochastic Lanczos Quadrature (SLQ) --
the same class of method used by PyHessian (Yao et al., 2020, cited in the
paper) and by Ghorbani et al. (2019), "An Investigation into Neural Net
Optimization via Hessian Eigenvalue Density".

Why this matters for the paper: reviewers #2/#3/#5 all raised the same
critique -- that tracking a single scalar (lambda_max) is an overly reductive
characterization of the Hessian spectrum. This script directly answers that
critique with the full eigenvalue density (or a close, standard approximation
of it), reusing the exact same HVP machinery already validated for lambda_max
and the Hutchinson trace, at a modest additional computational cost.

Method (matrix-free Lanczos + Gauss quadrature, Golub & Meurant framework):
  For each of `num_vectors` random starting vectors v0:
    1. Run `lanczos_steps` steps of the Lanczos algorithm using compute_hvp as
       the matrix-vector product operator (H is never explicitly formed).
       Full reorthogonalization is used at each step (cheap here: N ~ 422k,
       lanczos_steps ~50, trivially fast compared to the HVP calls themselves).
    2. This yields a small (lanczos_steps x lanczos_steps) tridiagonal matrix T.
    3. Eigen-decomposing T (trivial, small matrix) gives Ritz values theta_i
       (approximate eigenvalues of H) and quadrature weights tau_i^2 (squared
       first component of each eigenvector of T).
  Averaging (theta_i, tau_i^2) pairs over multiple random vectors gives an
  unbiased estimator of the full eigenvalue density of H (not just its max).

Usage:
    python spectral_density_slq.py --checkpoint checkpoints/ratio50_seed42/model_step_24500.pt \
        --train_ratio 0.5 --seed 42 --num_vectors 10 --lanczos_steps 50
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from generate_dataset import ModularArithmeticDataset
from model_architecture import ToyTransformer
from hessian_topology import compute_hvp, P_MODULO, BATCH_SIZE


def lanczos_tridiagonal(model: nn.Module, loss: torch.Tensor, total_params: int,
                         device: torch.device, num_steps: int = 50):
    """
    Runs matrix-free Lanczos tridiagonalization of the Hessian operator using
    compute_hvp as the matrix-vector product. Full reorthogonalization against
    all previous Lanczos vectors is used to counteract float32 orthogonality
    loss (standard practice for SLQ at this problem scale).

    Returns:
        alphas (list[float]): diagonal entries of the tridiagonal matrix T
        betas (list[float]):  off-diagonal entries of T (len(alphas)-1 entries)
    """
    v = torch.randn(total_params, device=device)
    v = v / torch.norm(v)
    v_prev = torch.zeros(total_params, device=device)
    beta_prev = 0.0

    V = [v.clone()]
    alphas, betas = [], []

    for j in range(num_steps):
        w = compute_hvp(model, loss, v)
        alpha = torch.dot(w, v).item()
        alphas.append(alpha)

        if j == num_steps - 1:
            break

        w = w - alpha * v - beta_prev * v_prev
        # Full reorthogonalization (cheap: num_steps x N, dominated by the HVP cost itself)
        for vi in V:
            w = w - torch.dot(w, vi) * vi

        beta = torch.norm(w).item()
        if beta < 1e-10:
            break  # Krylov subspace exhausted early (rare, harmless)
        betas.append(beta)

        v_prev = v
        v = w / beta
        V.append(v.clone())
        beta_prev = beta

    return alphas, betas


def slq_spectral_density(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor,
                          criterion: nn.Module, device: torch.device,
                          num_vectors: int = 10, lanczos_steps: int = 50):
    """
    Runs SLQ over `num_vectors` independent random Lanczos runs and pools the
    resulting (Ritz value, weight) pairs into a single set of quadrature nodes
    approximating the full eigenvalue density of H.
    """
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        sdp_context = sdpa_kernel(SDPBackend.MATH)
    except ImportError:
        sdp_context = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)

    with sdp_context:
        logits = model(inputs)
        loss = criterion(logits[:, -1, :], targets[:, -1])

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    all_ritz_values = []
    all_weights = []

    for run in range(num_vectors):
        alphas, betas = lanczos_tridiagonal(model, loss, total_params, device, num_steps=lanczos_steps)
        m = len(alphas)
        T = np.diag(alphas)
        for i in range(len(betas)):
            T[i, i+1] = betas[i]
            T[i+1, i] = betas[i]

        eigvals, eigvecs = np.linalg.eigh(T)
        weights = eigvecs[0, :] ** 2  # squared first component -> Gauss quadrature weight

        all_ritz_values.extend(eigvals.tolist())
        all_weights.extend((weights / num_vectors).tolist())

        print(f"  Lanczos run {run+1}/{num_vectors} | top Ritz value: {eigvals.max():.3f} | "
              f"actual steps taken: {m}")

    return all_ritz_values, all_weights, loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--train_ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_vectors", type=int, default=10, help="Number of independent Lanczos runs (random starting vectors).")
    parser.add_argument("--lanczos_steps", type=int, default=50, help="Krylov subspace dimension per run.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint} | ratio={args.train_ratio} | seed={args.seed} | "
          f"num_vectors={args.num_vectors} | lanczos_steps={args.lanczos_steps}\n")

    train_dataset = ModularArithmeticDataset(p=P_MODULO, split='train', train_ratio=args.train_ratio, seed=args.seed)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    x_batch, y_batch = next(iter(train_loader))
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

    model = ToyTransformer(vocab_size=train_dataset.vocab_size).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    ritz_values, weights, loss_val = slq_spectral_density(
        model, x_batch, y_batch, criterion, device,
        num_vectors=args.num_vectors, lanczos_steps=args.lanczos_steps
    )

    ritz_arr = np.array(ritz_values)
    weight_arr = np.array(weights)
    top_eigs = np.sort(ritz_arr)[::-1][:10]
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # E_v[v^T H v] over a unit-norm random v equals tr(H)/N (rotational invariance of the sphere),
    # so tr(H) ≈ N * mean_over_runs(sum_i theta_i * tau_i^2). The `weights` array already has the
    # per-run 1/num_vectors factor folded in, so summing directly gives the mean-over-runs quantity.
    slq_trace_estimate = total_params * np.sum(ritz_arr * weight_arr)

    print(f"\n=== Resumen ===")
    print(f"Top 10 Ritz values (candidatos a autovalores dominantes): {np.round(top_eigs, 2)}")
    print(f"Estimador de traza vía SLQ (para contraste con Hutchinson): {slq_trace_estimate:.3f}")

    out = {
        "checkpoint": args.checkpoint,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "num_vectors": args.num_vectors,
        "lanczos_steps": args.lanczos_steps,
        "loss_at_checkpoint": loss_val,
        "ritz_values": ritz_values,
        "weights": weights,
    }
    out_path = args.checkpoint.replace("/", "_").replace(".pt", "") + "_spectral_density.json"
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"\nGuardado en: {out_path}")


if __name__ == "__main__":
    main()
