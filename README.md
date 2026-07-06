# Beyond a Single Eigenvalue: A Multi-Seed Spectral Analysis of Grokking's Geometry

Code accompanying the paper on multi-seed replication of grokking's
behavioral and geometric signatures, submitted to TMLR.

## What changed from the original single-run study

The original version of this project tracked a single training run per
condition. This revision adds:

1. **Multi-seed replication** (4 seeds x 3 data-density conditions = 12 runs)
   of the core grokking experiment.
2. **A diagnostic protocol** isolating measurement noise (power-iteration
   initialization and mini-batch sampling) from genuine variation across
   trained models.
3. **Stochastic Lanczos Quadrature (SLQ)**, a convergence-robust alternative
   to power iteration that additionally recovers the full eigenvalue density
   rather than a single scalar.
4. **An instrumented, multi-seed optimizer ablation** that tracks the weight L2 norm
   alongside accuracy and lambda_max, identifying *why* Adam and SGD (both
   using coupled weight decay) behave near-identically: both collapse the
   weight norm by more than three orders of magnitude within the first
   ~1,500 steps, regardless of the underlying update rule. Replicated across
   all 4 seeds (`grokking_optimizer_ablation.py --seed <N>`); the qualitative
   outcome and the collapse mechanism both reproduce with very low
   seed-to-seed variance.
5. **A noise-disentangling control** (`measure_intrinsic_noise.py --fixed_batch`)
   isolating mini-batch sampling noise from power-iteration initialization
   noise in the repeated-measurement diagnostic, showing that mini-batch
   sampling is the dominant source of instability at most (though not all)
   of the checkpoints examined.

An earlier natural-language (TinyStories) validation experiment was removed
from this revision after a data-loading bug was discovered; see
`experimental/README.md` for details.

## Repository structure

```
.
├── generate_dataset.py              # Modular arithmetic task (a + b mod p)
├── model_architecture.py            # Small causal Transformer (~422K params)
├── train_and_grok.py                # Training loop (parameterized: --seed, --train_ratio, --weight_decay, --run_name)
├── hessian_topology.py              # lambda_max (power iteration) + Hutchinson trace, per checkpoint
├── grokking_optimizer_ablation.py   # AdamW vs. Adam vs. SGD ablation (Appendix A), with weight-norm tracking; --seed <N>, replicated across 4 seeds
├── visualize_paper.py               # Legacy single-run figure generator (original study)
│
├── run_all_experiments.py           # Orchestrates the 12-run multi-seed training grid
├── measure_intrinsic_noise.py       # Repeated power-iteration measurement on frozen checkpoints; --fixed_batch isolates init. vs. batch noise
├── spectral_density_slq.py          # Stochastic Lanczos Quadrature spectral density estimator
├── run_all_spectral_density.py      # Orchestrates the 36-checkpoint SLQ extraction grid
├── plot_spectral_density.py         # Exploratory spectral density visualization
│
├── figures/                         # Camera-ready figure generators for the paper (read directly from real output JSONs)
│   ├── generate_figure1_accuracy.py
│   ├── generate_figure2_measurement_noise.py
│   ├── generate_figure3_lambda_slq_distribution.py
│   ├── generate_figure4_spectral_density_grid.py
│   └── generate_figureA1_optimizer_ablation.py
│
├── experimental/                    # Retired TinyStories code; not used in the paper (see experimental/README.md)
│
├── requirements.txt
└── README_SETUP.md                  # Full installation and execution guide
```

## Quick start

See `README_SETUP.md` for full installation instructions (Python environment,
PyTorch with/without GPU, and step-by-step execution).

Minimal smoke test:
```bash
python3 train_and_grok.py --seed 1 --train_ratio 0.5 --max_steps 300 --run_name smoketest
python3 hessian_topology.py --run_name smoketest --train_ratio 0.5 --seed 1 --trace_samples 5
```

Full multi-seed replication (~8.5 hours on CPU, faster with a GPU):
```bash
python3 run_all_experiments.py
python3 run_all_spectral_density.py
```

## Data availability

All datasets are synthetic and generated on the fly by `generate_dataset.py`;
no external data download is required for the core experiments.
