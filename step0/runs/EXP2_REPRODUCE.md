# Experiment 2 (reliability / error detection) — reproduction

All commands from `~/code/outfitmatch-research/step0`, prefix `PYTHONPATH=./pylibs:./scripts`. Uses the frozen manifests and the frozen Phase-2 visual-MLP checkpoints (seeds 0-4); nothing is retrained except the supplementary linear baseline.

1. Pre-registration (written before any score was computed): `../EXPERIMENT2_PREREG.md`
2. Extract logits, kNN-pair distances (k in {10,50,200}), cost: `python3 scripts/60_extract_scores.py`  -> `runs/exp2/scores.npz`, `runs/exp2/cost.json`
3. Main comparison (AUROC, risk@80%, fixed val_id threshold, deployment view, risk-coverage, calibration, kNN-vs-year, subgroups, user cap, cost): `python3 scripts/61_exp2_main.py | tee results/exp2_main_report.txt` -> `runs/exp2/main_bootstrap.pkl`, `runs/exp2/figures/*.png`
4. Formal Gate 1 / Gate 2: `python3 scripts/62_exp2_gates.py | tee results/exp2_gates_report.txt` (needs step 3's pickle)
5. Supplement (run after the reliability results; not used by them): `bash scripts/run63.sh` -> `results/exp2_linear_baseline_report.txt`, `runs/exp2/linear/`. `results/exp2_linear_baseline_report_firstgrid.txt` is the first 4-config grid (kept for disclosure).

Libraries: `scripts/exp2_lib.py` (AUROC / risk-coverage / ECE / bootstrap), `scripts/exp2_data.py` (data, thresholds, meta detector). `taskb_lib.py` gained one extra `Net` variant (`linear`); existing variants are unchanged.
