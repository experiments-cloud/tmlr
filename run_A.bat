@echo off
REM run_A.bat -- Appendix B experiment. Run from a folder that contains:
REM   generate_dataset.py, model_architecture.py, hessian_topology.py (from your original folder)
REM   train_and_grok_ext.py, measure_curvature_seeded.py, run_A.bat (from kit_A.zip)
REM Resumable: finished trainings are skipped; measurement skips checkpoints already done.
REM Disk: about 220 MB of checkpoints per run, about 1.8 GB in total.
if not exist logs mkdir logs

echo Phase 1
if not exist grokking_telemetry_ext_ratio20_seed42.json (echo training ext_ratio20_seed42 & python3 train_and_grok_ext.py --seed 42 --train_ratio 0.2 --max_steps 25000 --run_name ext_ratio20_seed42 > logs\train_ext_ratio20_seed42.log 2>&1)
if not exist grokking_telemetry_ext_ratio20_seed1.json (echo training ext_ratio20_seed1 & python3 train_and_grok_ext.py --seed 1 --train_ratio 0.2 --max_steps 25000 --run_name ext_ratio20_seed1 > logs\train_ext_ratio20_seed1.log 2>&1)
if not exist grokking_telemetry_ext_ratio20_seed2.json (echo training ext_ratio20_seed2 & python3 train_and_grok_ext.py --seed 2 --train_ratio 0.2 --max_steps 25000 --run_name ext_ratio20_seed2 > logs\train_ext_ratio20_seed2.log 2>&1)
if not exist grokking_telemetry_ext_ratio20_seed3.json (echo training ext_ratio20_seed3 & python3 train_and_grok_ext.py --seed 3 --train_ratio 0.2 --max_steps 25000 --run_name ext_ratio20_seed3 > logs\train_ext_ratio20_seed3.log 2>&1)
if not exist grokking_telemetry_ext_ratio35_seed42.json (echo training ext_ratio35_seed42 & python3 train_and_grok_ext.py --seed 42 --train_ratio 0.35 --max_steps 25000 --run_name ext_ratio35_seed42 > logs\train_ext_ratio35_seed42.log 2>&1)
if not exist grokking_telemetry_ext_ratio35_seed1.json (echo training ext_ratio35_seed1 & python3 train_and_grok_ext.py --seed 1 --train_ratio 0.35 --max_steps 25000 --run_name ext_ratio35_seed1 > logs\train_ext_ratio35_seed1.log 2>&1)
if not exist grokking_telemetry_ext_ratio35_seed2.json (echo training ext_ratio35_seed2 & python3 train_and_grok_ext.py --seed 2 --train_ratio 0.35 --max_steps 25000 --run_name ext_ratio35_seed2 > logs\train_ext_ratio35_seed2.log 2>&1)
if not exist grokking_telemetry_ext_ratio35_seed3.json (echo training ext_ratio35_seed3 & python3 train_and_grok_ext.py --seed 3 --train_ratio 0.35 --max_steps 25000 --run_name ext_ratio35_seed3 > logs\train_ext_ratio35_seed3.log 2>&1)

echo Phase 2
echo measuring ext_ratio20_seed42 & python3 measure_curvature_seeded.py --repo . --run_name ext_ratio20_seed42 --train_ratio 0.2 --seed 42 > logs\curv_ext_ratio20_seed42.log 2>&1
echo measuring ext_ratio20_seed1 & python3 measure_curvature_seeded.py --repo . --run_name ext_ratio20_seed1 --train_ratio 0.2 --seed 1 > logs\curv_ext_ratio20_seed1.log 2>&1
echo measuring ext_ratio20_seed2 & python3 measure_curvature_seeded.py --repo . --run_name ext_ratio20_seed2 --train_ratio 0.2 --seed 2 > logs\curv_ext_ratio20_seed2.log 2>&1
echo measuring ext_ratio20_seed3 & python3 measure_curvature_seeded.py --repo . --run_name ext_ratio20_seed3 --train_ratio 0.2 --seed 3 > logs\curv_ext_ratio20_seed3.log 2>&1
echo measuring ext_ratio35_seed42 & python3 measure_curvature_seeded.py --repo . --run_name ext_ratio35_seed42 --train_ratio 0.35 --seed 42 > logs\curv_ext_ratio35_seed42.log 2>&1
echo measuring ext_ratio35_seed1 & python3 measure_curvature_seeded.py --repo . --run_name ext_ratio35_seed1 --train_ratio 0.35 --seed 1 > logs\curv_ext_ratio35_seed1.log 2>&1
echo measuring ext_ratio35_seed2 & python3 measure_curvature_seeded.py --repo . --run_name ext_ratio35_seed2 --train_ratio 0.35 --seed 2 > logs\curv_ext_ratio35_seed2.log 2>&1
echo measuring ext_ratio35_seed3 & python3 measure_curvature_seeded.py --repo . --run_name ext_ratio35_seed3 --train_ratio 0.35 --seed 3 > logs\curv_ext_ratio35_seed3.log 2>&1

echo Done
