# Installation and Execution Guide — Multi-Seed Replication

This package runs the full grid of **4 seeds x 3 data densities (50%/25%/10%) = 12 runs**,
each training for 25,000 steps and extracting the continuous spectral trajectory (lambda_max + Hutchinson trace).

---

## 1. Prerequisites

### 1.1 Python
You need **Python 3.9, 3.10, 3.11, or 3.12**. Check what you have installed:

```bash
python3 --version
```

If you don't have Python, or have a very old version (<3.9):
- **Windows**: download from https://www.python.org/downloads/ (check "Add Python to PATH" during installation).
- **macOS**: `brew install python@3.11` (requires Homebrew: https://brew.sh)
- **Linux (Ubuntu/Debian)**: `sudo apt update && sudo apt install python3 python3-venv python3-pip`

### 1.2 Do you have an NVIDIA GPU?
This determines which version of PyTorch to install. Check with:

```bash
nvidia-smi
```

- If the command **works** and shows a table with your GPU → you have an NVIDIA GPU available; follow the CUDA installation path below (faster).
- If it says "command not found" or doesn't recognize the hardware → you'll run on CPU (slower, but **fully viable** for this experiment — see time estimates below).

---

## 2. Create a virtual environment (recommended, avoids conflicts with other projects)

```bash
# Inside the folder containing these scripts
python3 -m venv venv

# Activate the environment:
# On Linux/macOS:
source venv/bin/activate
# On Windows (cmd):
venv\Scripts\activate.bat
# On Windows (PowerShell):
venv\Scripts\Activate.ps1
```

You'll know it's active because your terminal will show `(venv)` at the start of the line.

---

## 3. Install PyTorch

**Option A — You have an NVIDIA GPU (recommended if you do, ~10-20x faster):**

First check which CUDA version your driver supports (shown in the top-right corner of the `nvidia-smi` table, under "CUDA Version"). Then install the corresponding build. For CUDA 12.x (the most common as of 2025-2026):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

If your CUDA version is different, use the official selector to get the exact command:
https://pytorch.org/get-started/locally/

**Option B — CPU only (no GPU, or you'd rather not deal with CUDA drivers):**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Verify the installation:**

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

This should print the version and `True` or `False` depending on your hardware, with no errors.

---

## 4. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

(This installs `numpy`; you already installed PyTorch in the previous step with the command specific to your hardware.)

---

## 5. Verify that all files are in the same folder

```
your_folder/
├── generate_dataset.py
├── model_architecture.py
├── train_and_grok.py
├── hessian_topology.py
├── run_all_experiments.py
├── visualize_paper.py
├── requirements.txt
└── README_SETUP.md   (this file)
```

**Do not rename or move the files relative to each other** — `train_and_grok.py` and `hessian_topology.py` import directly from `generate_dataset.py` and `model_architecture.py`.

---

## 6. Disk space required

Each run saves ~66 model checkpoints (about 1.7 MB each, 422K-parameter model) →
~110 MB per run x 12 runs ≈ **1.3 GB total**. Make sure you have at least 2 GB free.

---

## 7. Running the experiments

### 7.1 First, a quick sanity check (recommended, takes ~1 minute)

Before launching the full 12 runs, validate that everything works with a tiny run:

```bash
python3 train_and_grok.py --seed 99 --train_ratio 0.5 --max_steps 300 --run_name smoketest
python3 hessian_topology.py --run_name smoketest --train_ratio 0.5 --seed 99 --trace_samples 5
```

If both finish without errors and print lambda_max and tr(H) numbers, you're ready. Clean up afterward:

```bash
# Linux/macOS
rm -rf checkpoints/smoketest grokking_telemetry_smoketest*.json
# Windows (PowerShell)
Remove-Item -Recurse -Force checkpoints\smoketest, grokking_telemetry_smoketest*.json
```

### 7.2 See the full plan without running anything (dry run)

```bash
python3 run_all_experiments.py --dry_run
```

This prints the 12 combinations that will be run, without spending any compute.

### 7.3 Launch the full grid

```bash
python3 run_all_experiments.py
```

This runs the 12 combinations **sequentially and automatically**: trains, then extracts the spectrum,
then moves to the next one. Each run generates its own log in `logs/train_<run>.log` and `logs/hessian_<run>.log`,
and its own telemetry in `grokking_telemetry_<run_name>_with_hessian.json`.

### 7.4 If it gets interrupted midway (terminal closes, power outage, etc.)

**No problem.** Just run the exact same command again:

```bash
python3 run_all_experiments.py
```

The script automatically detects which runs already finished (checks whether their
`grokking_telemetry_<run_name>_with_hessian.json` file exists with valid data)
and **skips the ones already done**, resuming only what's pending.

### 7.5 Running it in the background (optional, useful if you're going to close the terminal)

**Linux/macOS:**
```bash
nohup python3 run_all_experiments.py > run_all.log 2>&1 &
```
You can close the terminal and it will keep running. To check progress:
```bash
tail -f run_all.log
```

**Windows (PowerShell):** open a window and leave it minimized, or use Task Scheduler if you need it to survive a logout. The simplest approach is to just leave the PowerShell window open and minimized.

---

## 8. Time estimates

Measured on a CPU-only environment (no GPU) equivalent to a modern desktop processor:

| Phase | Time per run | Total time (12 runs) |
|---|---|---|
| Training (25,000 steps) | ~35 min | ~7 hours |
| Spectral extraction (66 checkpoints, lambda_max + trace) | ~7 min | ~1.4 hours |
| **Total** | **~42 min** | **~8.5 hours** |

With an NVIDIA GPU (even a modest laptop-class RTX 30/40 series), expect a
**5-10x** reduction in training time (spectral extraction via HVP benefits less,
since it's limited by the number of double backward passes, not batch size).

**Practical recommendation:** launch it overnight or while doing something else; it doesn't
require you to be in front of the screen. With `nohup` (Linux/macOS) you can close the
terminal without issue.

---

## 9. Which files do I care about at the end?

For each of the 12 runs:

- `grokking_telemetry_<run_name>_with_hessian.json` → contains `steps`, `train_loss`, `val_loss`,
  `val_accuracy`, `checkpoint_steps`, `lambda_max`, `hessian_trace`, and metadata (`seed`, `train_ratio`, `weight_decay`).

These 12 files are what's needed for the statistical aggregation (mean ± standard deviation,
regenerating figures with confidence bands). Once the grid finishes, upload those 12
JSON files to continue with the analysis.

---

## 10. Common issues

| Symptom | Likely cause | Solution |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` | The virtual environment isn't activated, or the installation failed | Check for `(venv)` in the prompt; reinstall with the command from step 3 |
| `CUDA out of memory` | Very unlikely with this model (422K params), but if it happens | Try a lower `--max_steps` to test, or force CPU with the environment variable `CUDA_VISIBLE_DEVICES=""` before the command |
| Training is much slower than estimated | Running on CPU with few cores, or other heavy processes are open | Check with Task Manager/`htop`; close unnecessary programs |
| `FileNotFoundError` when running `hessian_topology.py` | You didn't run `train_and_grok.py` first for that same `--run_name` | Run training first; `run_all_experiments.py` already does this in the correct order automatically |
