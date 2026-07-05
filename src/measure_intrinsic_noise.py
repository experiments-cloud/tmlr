"""
measure_intrinsic_noise.py

Diagnostic experiment: for a FIXED, already-trained checkpoint (frozen weights),
repeatedly re-measure lambda_max and the Hutchinson trace using fresh random
batches and fresh random power-iteration init vectors each time.

Purpose: separate MEASUREMENT NOISE (same weights, different random batch/vector)
from REAL INTER-SEED VARIANCE (different trained weights, as already observed
across the 4-seed replication). If measurement noise alone explains a large
fraction of the spread seen across seeds, that is an important, honest,
reportable finding about the reliability of single-point HVP/power-iteration
spectral estimates -- independent of whether the underlying geometry itself
varies across training runs.

This script does NOT retrain anything. It only loads existing .pt checkpoints
and repeatedly runs the same spectral extraction already used in
hessian_topology.py. Cheap: a handful of forward+double-backward passes per
repetition, no full optimization loop.

Usage example (run from the same folder as your checkpoints/ directory):

    python measure_intrinsic_noise.py \
        --checkpoint checkpoints/ratio50_seed42/model_step_24500.pt \
        --train_ratio 0.5 --seed 42 --repeats 10

Run this for a handful of checkpoints spanning:
  (a) different ratios (50/25/10) at a LATE step (e.g. the last saved checkpoint), and
  (b) optionally the SAME run at an EARLY/transition step (e.g. step ~500-1000),
to see whether measurement noise differs by training phase.
"""

import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from generate_dataset import ModularArithmeticDataset
from model_architecture import ToyTransformer
from hessian_topology import get_dominant_eigenvalue, P_MODULO, BATCH_SIZE, NUM_POWER_ITERATIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a single model_step_*.pt file.")
    parser.add_argument("--train_ratio", type=float, required=True, help="train_ratio used to train this checkpoint.")
    parser.add_argument("--seed", type=int, required=True, help="seed used to train this checkpoint (for dataset reconstruction only).")
    parser.add_argument("--repeats", type=int, default=10, help="Number of independent re-measurements on the SAME frozen weights.")
    parser.add_argument("--trace_samples", type=int, default=10)
    parser.add_argument("--num_iterations", type=int, default=NUM_POWER_ITERATIONS,
                         help=f"Power iteration steps per measurement (paper/original default: {NUM_POWER_ITERATIONS}). "
                              f"Increase this (e.g. 100) to test whether more iterations reduce measurement noise/bimodal non-convergence.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint} | ratio={args.train_ratio} | seed={args.seed} | "
          f"repeats={args.repeats} | power_iterations={args.num_iterations}\n")

    train_dataset = ModularArithmeticDataset(p=P_MODULO, split='train', train_ratio=args.train_ratio, seed=args.seed)
    # shuffle=True is essential here: each call to next(iter(...)) below draws a genuinely different random batch
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = ToyTransformer(vocab_size=train_dataset.vocab_size).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    criterion = nn.CrossEntropyLoss()

    lmax_results = []
    trace_results = []

    for i in range(args.repeats):
        # Fresh random batch each repetition (same frozen weights, different sampled data)
        x_batch, y_batch = next(iter(train_loader))
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # get_dominant_eigenvalue also draws a fresh random init vector v0 internally each call,
        # and fresh random Rademacher vectors for the trace estimator.
        lmax, trace, _ = get_dominant_eigenvalue(
            model, x_batch, y_batch, criterion, device,
            num_iterations=args.num_iterations, num_trace_samples=args.trace_samples
        )
        print(f"  Repeat {i+1:2d}/{args.repeats} | lambda_max = {lmax:10.3f} | tr(H) = {trace:10.3f}")
        lmax_results.append(lmax)
        trace_results.append(trace)

    import statistics
    print(f"\n=== Resultado: {args.repeats} mediciones repetidas SOBRE LOS MISMOS PESOS ===")
    print(f"lambda_max: media={statistics.mean(lmax_results):.3f}  std={statistics.pstdev(lmax_results):.3f}  "
          f"min={min(lmax_results):.3f}  max={max(lmax_results):.3f}")
    print(f"tr(H):      media={statistics.mean(trace_results):.3f}  std={statistics.pstdev(trace_results):.3f}  "
          f"min={min(trace_results):.3f}  max={max(trace_results):.3f}")

    out = {
        "checkpoint": args.checkpoint,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "repeats": args.repeats,
        "num_iterations": args.num_iterations,
        "lambda_max_samples": lmax_results,
        "trace_samples": trace_results,
    }
    out_path = args.checkpoint.replace("/", "_").replace(".pt", "") + f"_iters{args.num_iterations}_noise_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nGuardado en: {out_path}")


if __name__ == "__main__":
    main()
