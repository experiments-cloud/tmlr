# Revision materials (TMLR revision)

The scripts that produced the results of the original submission (`train_and_grok.py`, `hessian_topology.py`, `spectral_density_slq.py`, `measure_intrinsic_noise.py`, `model_architecture.py`, `generate_dataset.py`, `run_all_experiments.py`, `run_all_spectral_density.py`, `grokking_optimizer_ablation.py`) are **unchanged** in this revision. The files below were added for the revision.

## Exact objective measured by all Hessian computations
- **Loss:** mean cross-entropy of the logits at the **final input position** ("=") against the answer token `c`. The other shifted targets returned by the data loader are not used. The same objective is used for training, accuracy and every Hessian.
- **Mode:** `model.eval()` (dropout off); training uses dropout 0.1.
- **Regularization:** the weight-decay term is not included in the measured Hessian.
- **Batches:** 512 training examples.
- **Attention:** unfused (MATH) kernels.

## New experiments
| Script | Paper | What it does |
|---|---|---|
| `controlled_pi_vs_slq.py` | Section 5.3, Table 4 | Power iteration (20 and 100 iterations) and Lanczos/SLQ on **identical** mini-batch Hessians (8 seeded batches); fixed-Hessian convergence check with Ritz residuals; full-training-set reference. |
| `train_and_grok_ext.py` | Appendix B | Same training as `train_and_grok.py` (identical trajectory, verified bit-for-bit on the first 400 steps). It also logs loss and accuracy on the **full training set** without consuming random numbers, and saves checkpoints every 100 steps up to step 10,000. |
| `measure_curvature_seeded.py` | Appendix B, Table 9, Figure 7 | At each checkpoint: top eigenvalue on 4 seeded batches (the same batches at every checkpoint), and λ1, λ2 of the full-training-set Hessian with residuals. |
| `run_A.bat` | Appendix B | Runs the 8 Appendix B runs (20% and 35% density, seeds 42/1/2/3) and their curvature measurements. |

## Analysis and figures
- `revision/analyze_existing.py`: recomputes every number in Tables 2, 3, 5 and 6 from the original result JSONs.
- `revision/summarize_controlled.py`: summarizes the outputs of `controlled_pi_vs_slq.py` (Table 4).
- `revision/analyze_expA.py`: produces Table 9 and Figure 7.
- `revision/make_new_figures.py`: produces Figures 3 and 5.
- `revision/verify_appendix.py`: checks the fixed-batch column of Table 3, Table 7 and Table 8.
- `figures/generate_figure4_slq_boxplots.py`: produces Figure 4 (panel labels early/intermediate/late).
- `figures/generate_figure{1,2,A1}_*.py`: now use the **sample** standard deviation (`ddof=1`), as the revised paper does throughout.

## Raw outputs
`revision/results/` contains:
- the raw outputs of Section 5.3 (`ctrl_*.json`, `summary_controlled.json`). These were produced with an earlier revision of `controlled_pi_vs_slq.py`, identical in computation, that did not yet record the `torch_version` and `threads` fields and did not have the import fallback for PyTorch < 2.3;
- the raw outputs of Appendix B (`grokking_telemetry_ext_*.json`, `curvature_ext_*.json`).

## Result files of the original experiments (`results/`)
These are the raw outputs behind the original tables and figures:
- `results/main_grid/`: the 12 training runs (`grokking_telemetry_ratio*_seed*_with_hessian.json`: validation accuracy, logged training loss, power-iteration λmax and trace per checkpoint). Used for Tables 2 and 6 and Figures 1 and 3.
- `results/slq/`: 56 SLQ extractions, i.e. 36 at steps 500/2,000/24,500 plus 20 at the finer steps for 25% density (pooled Ritz values and weights). Used for Tables 5, 6 and 7 and Figures 4 and 5.
- `results/noise_diagnostic/`: the 14 repeated-measurement diagnostics. Used for Table 3 and Figure 2.
- `results/optimizer_ablation/`: the 4 per-seed ablation runs. Used for Table 8 and Figure 6.

## Retrained checkpoints used in Section 5.3 (`revision/checkpoints_retrained/`)
The original checkpoint files were not retained, and training is not bit-for-bit reproducible, so the four retrained checkpoints analysed in Section 5.3 are provided for exact reproduction of Table 4:

```
b458c663b473ca7b06c7a4b1f74730bee50f97d13204de73c77b40bccc196449  revision/checkpoints_retrained/ratio10_seed42/model_step_2000.pt
218765d4957438a339b6149b4ccf2976db377fa4f3aaad13e6b99ed1151306a7  revision/checkpoints_retrained/ratio10_seed42/model_step_24500.pt
0e771cfa36e01083b797994a80e4c4ad25ab6a1d45f64252df4771ad415e19de  revision/checkpoints_retrained/ratio25_seed42/model_step_2000.pt
c64cea0ccdb22df932e0890210b02e556b7c492aa2ea497a4d3b53e542536f75  revision/checkpoints_retrained/ratio25_seed42/model_step_24500.pt
```

## Reproducibility note
Runs are reproducible statistically, not bit-for-bit: floating-point differences accumulate during training. The checkpoints of Section 5.3 are retrained seed-42 runs, not the original checkpoint files.

## Note on standard deviations
All coefficients of variation reported in the paper use the sample standard deviation (n - 1). `revision/summarize_controlled.py` and `revision/analyze_expA.py` compute them that way. The `batch_sd` field stored in the `curvature_ext_*.json` files by `measure_curvature_seeded.py` is the population standard deviation and is not used for any reported number; the analysis recomputes the spread from `batch_top`.
