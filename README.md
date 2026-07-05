# Reliability of Hessian Spectral Diagnostics in Grokking: A Multi-Seed Replication

This repository contains the code and raw experimental data supporting the paper
submitted to Transactions on Machine Learning Research (TMLR). It accompanies a
multi-seed replication and methodological re-examination of Hessian-based spectral
diagnostics (dominant eigenvalue, trace, and eigenvalue density) applied to delayed
generalization ("grokking") on a modular-addition task.

**Note for reviewers:** this repository is anonymized for double-blind review. No
author names, institutional affiliations, or identifying information appear anywhere
in the code, comments, or commit history.

---

## Repository structure

```
.
├── src/                          Core scripts: dataset, model, training, spectral extraction
│   ├── generate_dataset.py           Modular-addition dataset (p=97, configurable train_ratio)
│   ├── model_architecture.py         ~422K-parameter causal Transformer
│   ├── train_and_grok.py             Training loop (parametrized by --seed, --train_ratio)
│   ├── hessian_topology.py           lambda_max (power iteration) + trace (Hutchinson) extraction
│   ├── measure_intrinsic_noise.py    Repeated-measurement diagnostic (Section 5.2)
│   ├── spectral_density_slq.py       Stochastic Lanczos Quadrature (Section 5.3)
│   ├── grokking_optimizer_ablation.py  AdamW vs. Adam vs. SGD ablation (Appendix A)
│   ├── run_all_experiments.py        Orchestrator: 4 seeds x 3 data densities (12 runs)
│   └── run_all_spectral_density.py   Orchestrator: SLQ across the 36-run grid
│
├── analysis/                     Figure-generation scripts (read data/, produce PDF+PNG)
│   ├── generate_figure1_accuracy.py
│   ├── generate_figure2_measurement_noise.py
│   ├── generate_figure3_lambda_slq_distribution.py
│   ├── generate_figure4_spectral_density_grid.py
│   └── generate_figureA1_optimizer_ablation.py
│
├── data/                         Raw experimental outputs (JSON), as reported in the paper
│   ├── telemetry/                    12 files: 4 seeds x 3 data densities (Table 3, Figure 1)
│   ├── noise_diagnostics/            10 files: repeated-measurement diagnostic (Table 4, Figure 2)
│   ├── spectral_density/             36 files: SLQ grid, 4 seeds x 3 densities x 3 phases (Table 5, Figures 3-4)
│   └── optimizer_ablation/           1 file: AdamW/Adam/SGD comparison (Appendix A)
│
└── requirements.txt
```

---

## Reproducing the results from scratch

### 1. Environment

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

GPU is not required; all experiments were run on CPU. A GPU (if `torch.cuda.is_available()`)
is used automatically if present.

### 2. Train the main experimental grid (12 runs)

```bash
cd src/
python run_all_experiments.py            # ~8-9 hours on CPU; resumable if interrupted
python run_all_experiments.py --dry_run  # to preview the run plan without executing
```

This populates `checkpoints/<run_name>/` with model weights and produces
`grokking_telemetry_<run_name>.json` per run (pre-Hessian). Running
`hessian_topology.py` for each run name (also invoked automatically inside
`run_all_experiments.py`) produces the final `..._with_hessian.json` files matching
those already provided in `data/telemetry/`.

### 3. Run the measurement-noise diagnostic (Section 5.2)

Requires the checkpoints from step 2. Example:

```bash
python measure_intrinsic_noise.py \
    --checkpoint checkpoints/ratio10_seed42/model_step_24500.pt \
    --train_ratio 0.1 --seed 42 --repeats 10 --num_iterations 20

python measure_intrinsic_noise.py \
    --checkpoint checkpoints/ratio10_seed42/model_step_24500.pt \
    --train_ratio 0.1 --seed 42 --repeats 10 --num_iterations 100
```

### 4. Run the SLQ spectral density grid (Section 5.3)

```bash
python run_all_spectral_density.py            # ~30-40 min; resumable
```

### 5. Run the optimizer ablation (Appendix A)

```bash
python grokking_optimizer_ablation.py         # ~70-80 min, trains AdamW/Adam/SGD sequentially
```

### 6. Regenerate the figures from the data already in `data/`

```bash
cd ../analysis/
python generate_figure1_accuracy.py --input_dir ../data/telemetry
python generate_figure2_measurement_noise.py --input_dir ../data/noise_diagnostics
python generate_figure3_lambda_slq_distribution.py --input_dir ../data/spectral_density
python generate_figure4_spectral_density_grid.py --input_dir ../data/spectral_density
python generate_figureA1_optimizer_ablation.py --input ../data/optimizer_ablation/optimizer_ablation_results.json
```

Each script writes both a `.png` and a `.pdf` version, and prints an explicit warning
if any expected seed or checkpoint is missing from the input data, rather than
silently proceeding with an incomplete grid.

---

## Data provenance and integrity notes

- All numerical claims in the paper are computed directly from the JSON files in
  `data/`; no reported number is manually transcribed or estimated from a figure.
- The `data/noise_diagnostics/` files without `iters100` in their filename were
  produced under the original 20-iteration power-iteration budget; those with
  `iters100` are the corresponding re-test at a 100-iteration budget (Section 5.2,
  Table 4).
- The `excluded_tinystories_experiment/` folder documents a discarded experiment and
  is not referenced by any figure or table; see its own `README.md`.

## Requirements

See `requirements.txt`. Core dependencies: `torch>=2.1.2`, `numpy`, `scipy`,
`matplotlib`.
