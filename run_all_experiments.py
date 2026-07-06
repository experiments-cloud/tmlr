"""
run_all_experiments.py

Orchestrator for the multi-seed replication study.

Runs the full grid: 4 seeds x 3 data-density ratios (50% / 25% / 10%) = 12 runs.
For each run: (1) trains the model and saves telemetry + checkpoints, then
(2) extracts the continuous Hessian spectral trajectory (lambda_max + Hutchinson trace).

Designed to be resumable: if interrupted, re-running this script will SKIP any
run whose final telemetry file already exists, so you do not lose completed work.

Usage:
    python run_all_experiments.py                  # runs the full grid
    python run_all_experiments.py --dry_run         # just prints the plan, runs nothing
    python run_all_experiments.py --max_steps 5000  # override step count (e.g. for a quick pilot)
"""

import argparse
import os
import subprocess
import sys
import time
import json

SEEDS = [42, 1, 2, 3]              # 42 = original run from the paper; 1,2,3 = new replicates
RATIOS = [0.5, 0.25, 0.10]         # relative abundance / moderate scarcity / severe scarcity
TRACE_SAMPLES = 10

def run_name_for(ratio: float, seed: int) -> str:
    return f"ratio{int(ratio*100)}_seed{seed}"

def already_done(run_name: str) -> bool:
    """A run counts as done if its post-Hessian telemetry file already exists and is valid JSON."""
    path = f"grokking_telemetry_{run_name}_with_hessian.json"
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            data = json.load(f)
        return len(data.get("lambda_max", [])) > 0
    except Exception:
        return False

def run_command(cmd: list, log_path: str) -> int:
    print(f"  $ {' '.join(cmd)}")
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=25000)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)

    plan = [(ratio, seed) for ratio in RATIOS for seed in SEEDS]

    print(f"Planned grid: {len(plan)} runs (seeds={SEEDS}, ratios={RATIOS}, max_steps={args.max_steps})\n")

    if args.dry_run:
        for ratio, seed in plan:
            rn = run_name_for(ratio, seed)
            status = "SKIP (already done)" if already_done(rn) else "TO RUN"
            print(f"  - {rn:20s} ratio={ratio:<5} seed={seed:<3} -> {status}")
        return

    start_all = time.time()
    for i, (ratio, seed) in enumerate(plan, start=1):
        run_name = run_name_for(ratio, seed)
        print(f"\n[{i}/{len(plan)}] Run: {run_name}")

        if already_done(run_name):
            print(f"  Already completed (found grokking_telemetry_{run_name}_with_hessian.json). Skipping.")
            continue

        t0 = time.time()

        # --- Step 1: training ---
        train_log = f"logs/train_{run_name}.log"
        rc = run_command(
            ["python3", "train_and_grok.py",
             "--seed", str(seed),
             "--train_ratio", str(ratio),
             "--max_steps", str(args.max_steps),
             "--run_name", run_name],
            train_log
        )
        if rc != 0:
            print(f"  !! Training FAILED (see {train_log}). Skipping spectral extraction for this run.")
            continue

        # --- Step 2: spectral extraction (lambda_max + Hutchinson trace) ---
        hess_log = f"logs/hessian_{run_name}.log"
        rc = run_command(
            ["python3", "hessian_topology.py",
             "--run_name", run_name,
             "--train_ratio", str(ratio),
             "--seed", str(seed),
             "--trace_samples", str(TRACE_SAMPLES)],
            hess_log
        )
        if rc != 0:
            print(f"  !! Spectral extraction FAILED (see {hess_log}).")
            continue

        elapsed = time.time() - t0
        print(f"  Done in {elapsed/60:.1f} min.")

    total_elapsed = time.time() - start_all
    print(f"\nAll done. Total elapsed: {total_elapsed/60:.1f} min ({total_elapsed/3600:.2f} h).")
    print("Per-run telemetry files: grokking_telemetry_<run_name>_with_hessian.json")


if __name__ == "__main__":
    main()
