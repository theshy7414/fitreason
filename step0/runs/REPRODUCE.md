# Reproduce Task B Phase 1 / Phase 2 (all from step0/, WSL Ubuntu-22.04, RTX 3070 Ti, torch 2.8.0+cu128, python 3.10.12)

Prerequisites (already done, deterministic): frozen manifests `manifests/` (sha256 in `manifests/MANIFEST_SHA256.txt`),
features `data/features_v1/` (`scripts/30_stream_extract_features.py`, `--finalize`), audit flags (`scripts/41_neardup_audit.py`).

```bash
export PYTHONPATH=./pylibs:./scripts
python3 scripts/51_instance_table.py           # per-instance baselines + audit flags -> results/exp1_instance_table_taskB.npz
python3 scripts/50_phase1.py                   # integrity checks, 4-config grid on val_id NLL, controls, test_id once, cost  -> runs/phase1/
python3 scripts/52_phase2_train.py             # visual & visual_cat x seeds 0-4, config from runs/phase1/selected_config.json -> runs/phase2/<variant>_seed<k>/
python3 scripts/53_phase2_report.py            # tables + figures -> results/exp1_phase2_report.txt, runs/phase2/figures/
```

Selected config (val_id NLL only, seed 0): variant visual, hidden 256, dropout 0.3, AdamW lr 3e-4, weight decay 1e-2, batch 256, <=40 epochs,
early stopping patience 5 on val_id NLL. Same config is used for visual_cat (not re-tuned). Seeds control init, data order and online negative sampling.
Per run directory: `best.pt` (checkpoint at best val NLL), `log.jsonl` (per-epoch curves), `run_info.json` (config, seed, epochs, params, time), `preds.npz` (logits), `env.json`.
Determinism check: Phase 2 `visual_seed0` reproduced the Phase 1 selected run exactly (val 59.7, test_id 62.3).
Note: shifted-year sets were never loaded in Phase 1 and never used for any selection.
