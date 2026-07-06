"""
run_all_spectral_density.py

Orchestrator for the SLQ spectral density extraction across the full grid:
4 seeds x 3 data-density ratios x 3 training phases (early/transition/late) = 36 runs.

Does NOT retrain anything -- it only re-measures already-trained checkpoints
(the same ones used for train_and_grok.py / hessian_topology.py / measure_intrinsic_noise.py).
Requires that checkpoints/<run_name>/model_step_<STEP>.pt already exist on disk.

Resumable: re-running this script skips any (ratio, seed, step) combination whose
output JSON already exists, so an interruption does not lose completed work.

Usage:
    python run_all_spectral_density.py                # runs the full 36-combo grid
    python run_all_spectral_density.py --dry_run       # just prints the plan, runs nothing
    python run_all_spectral_density.py --steps 500 24500   # override which phases to run
    python run_all_spectral_density.py --seeds 42           # e.g. only the original seed
"""

import argparse
import os
import subprocess
import sys
import time

SEEDS = [42, 1, 2, 3]
RATIOS = [0.5, 0.25, 0.10]
STEPS = [500, 2000, 24500]   # early / transition / late training phases
NUM_VECTORS = 10
LANCZOS_STEPS = 50


def run_name_for(ratio: float, seed: int) -> str:
    return f"ratio{int(ratio*100)}_seed{seed}"


def checkpoint_path(ratio: float, seed: int, step: int) -> str:
    return f"checkpoints/{run_name_for(ratio, seed)}/model_step_{step}.pt"


def output_json_path(ckpt_path: str) -> str:
    # Mirrors the naming logic inside spectral_density_slq.py exactly.
    return ckpt_path.replace("/", "_").replace(".pt", "") + "_spectral_density.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--ratios", type=float, nargs="+", default=RATIOS)
    parser.add_argument("--steps", type=int, nargs="+", default=STEPS)
    parser.add_argument("--num_vectors", type=int, default=NUM_VECTORS)
    parser.add_argument("--lanczos_steps", type=int, default=LANCZOS_STEPS)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)

    plan = [(ratio, seed, step) for ratio in args.ratios for seed in args.seeds for step in args.steps]
    print(f"Planned grid: {len(plan)} SLQ extractions "
          f"(seeds={args.seeds}, ratios={args.ratios}, steps={args.steps})\n")

    n_missing_ckpt = 0
    n_skip = 0
    n_todo = 0

    for ratio, seed, step in plan:
        ckpt = checkpoint_path(ratio, seed, step)
        out = output_json_path(ckpt)
        if not os.path.exists(ckpt):
            status = "!! CHECKPOINT NOT FOUND"
            n_missing_ckpt += 1
        elif os.path.exists(out):
            status = "SKIP (already done)"
            n_skip += 1
        else:
            status = "TO RUN"
            n_todo += 1
        print(f"  - ratio={int(ratio*100):<4}seed={seed:<4}step={step:<7} -> {status}")

    print(f"\nSummary: {n_todo} to run, {n_skip} already done, {n_missing_ckpt} missing checkpoints.")

    if args.dry_run:
        return

    if n_missing_ckpt > 0:
        print("\nWARNING: some checkpoints are missing. Those combinations will be skipped automatically.")

    start_all = time.time()
    completed, failed, skipped = 0, 0, 0

    for i, (ratio, seed, step) in enumerate(plan, start=1):
        ckpt = checkpoint_path(ratio, seed, step)
        out = output_json_path(ckpt)
        label = f"ratio{int(ratio*100)}_seed{seed}_step{step}"

        if not os.path.exists(ckpt):
            print(f"[{i}/{len(plan)}] {label}: checkpoint missing, skipping.")
            skipped += 1
            continue
        if os.path.exists(out):
            print(f"[{i}/{len(plan)}] {label}: already done, skipping.")
            skipped += 1
            continue

        print(f"[{i}/{len(plan)}] {label}: running SLQ...")
        t0 = time.time()
        log_path = f"logs/slq_{label}.log"
        cmd = ["python3", "spectral_density_slq.py",
               "--checkpoint", ckpt,
               "--train_ratio", str(ratio),
               "--seed", str(seed),
               "--num_vectors", str(args.num_vectors),
               "--lanczos_steps", str(args.lanczos_steps)]
        with open(log_path, "w") as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)

        elapsed = time.time() - t0
        if proc.returncode != 0:
            print(f"  !! FAILED (see {log_path})")
            failed += 1
        else:
            print(f"  Done in {elapsed:.1f}s.")
            completed += 1

    total_elapsed = time.time() - start_all
    print(f"\nAll done. Completed: {completed} | Failed: {failed} | Skipped: {skipped}")
    print(f"Total elapsed: {total_elapsed/60:.1f} min.")
    print("Output files: checkpoints_<run_name>_model_step_<STEP>_spectral_density.json")


if __name__ == "__main__":
    main()
