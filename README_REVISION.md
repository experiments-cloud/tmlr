# Revision materials (TMLR revision)

The files below were added for the revision.

## Exact objective measured by all Hessian computations
- **Loss:** mean cross-entropy of the logits at the **final input position** ("=") against the answer token `c`. The other shifted targets returned by the data loader are not used. The same objective is used for training, accuracy and every Hessian.
- **Mode:** `model.eval()` (dropout off); training uses dropout 0.1.
- **Regularization:** the weight-decay term is not included in the measured Hessian.
- **Batches:** 512 training examples.
- **Attention:** unfused (MATH) kernels.

## New experiments
| Script | Section | Description |
|---|---|---|
| `controlled_pi_vs_slq.py` | Section 5.3, Table 4 | Power iteration (20 and 100 iterations) and Lanczos/SLQ on **identical** mini-batch Hessians (8 seeded batches); fixed-Hessian convergence check with Ritz residuals; full-training-set reference. |
| `train_and_grok_ext.py` | Appendix B | Same training as `train_and_grok.py` (identical trajectory, verified bit-for-bit on the first 400 steps). It also logs loss and accuracy on the **full training set** without consuming random numbers, and saves checkpoints every 100 steps up to step 10,000. |
| `measure_curvature_seeded.py` | Appendix B, Table 9, Figure 7 | At each checkpoint: top eigenvalue on 4 seeded batches (the same batches at every checkpoint), and λ1, λ2 of the full-training-set Hessian with residuals. |
| `run_A.bat` | Appendix B | Runs the 8 Appendix B runs (20% and 35% density, seeds 42/1/2/3) and their curvature measurements. |

## Figures
- `figures/generate_figure{1,2,A1}_*.py`: now use the **sample** standard deviation (`ddof=1`), as the revised paper does throughout.

## Result files of the original experiments (`results/`)
These are the raw outputs behind the original tables and figures:
- `results/main_grid/`: the 12 training runs (`grokking_telemetry_ratio*_seed*_with_hessian.json`: validation accuracy, logged training loss, power-iteration λmax and trace per checkpoint). Used for Tables 2 and 6 and Figures 1 and 3.
- `results/slq/`: 56 SLQ extractions, i.e. 36 at steps 500/2,000/24,500 plus 20 at the finer steps for 25% density (pooled Ritz values and weights). Used for Tables 5, 6 and 7 and Figures 4 and 5.
- `results/noise_diagnostic/`: the 14 repeated-measurement diagnostics. Used for Table 3 and Figure 2.
- `results/optimizer_ablation/`: the 4 per-seed ablation runs. Used for Table 8 and Figure 6.

## Reproducibility note
Runs are reproducible statistically, not bit-for-bit: floating-point differences accumulate during training. The checkpoints analysed in Section 5.3 come from retrained seed-42 runs, not from the original checkpoint files.
