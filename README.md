# OutfitMatch Research — SHIFT15M temporal reliability

## 狀態（繁體中文）

- **研究線已暫停／結案（2026-10-13），結論為 bounded negative result**：在 SHIFT15M 上，跨年份（temporal shift）下沒有找到值得採用的新 reliability 方法；詳見 [`FINAL_RESEARCH_REPORT.md`](FINAL_RESEARCH_REPORT.md)（完整報告）與 [`FINAL_REPORT_SUMMARY_FOR_GPT.md`](FINAL_REPORT_SUMMARY_FOR_GPT.md)（精簡摘要）。
- 實驗規格／預註冊／狀態：`EXPERIMENT1_SPEC.md`、`EXPERIMENT2_PREREG.md`、`EXPERIMENT2_STATUS_REPORT.md`；報告用圖與驗算腳本在 `final_report_assets/`。
- 這個 repo **只含程式碼與文件**（腳本、文件、manifests、結果文字檔、小型 json、`step0/runs/exp2/scores.npz`（4.8 MB）、圖）。各 seed 的 `preds.npz`、`exp1_instance_table_taskB.npz` 未納入（在本機與備份中）。

### 大檔「不在 git」裡，在哪裡找

| 內容 | 本機位置 | 備註 |
|---|---|---|
| 特徵、原始資料（約 42 GB） | `/home/chi/outfitmatch-research/step0/data/` | 已被 `.gitignore` 排除 |
| 第三方 Python 套件 | `step0/pylibs/` | 排除 |
| 模型 checkpoint（`*.pt`） | `step0/runs/**/best.pt` | 排除 |
| bootstrap pickle | `step0/runs/exp2/main_bootstrap.pkl`（5.8 MB） | 排除 |
| near-duplicate flags（7 MB） | `step0/results/exp1_neardup_flags_taskB.json` | 排除（> 5 MB） |
| **完整備份（約 42 GB）** | `/home/chi/backups/outfitmatch-research-20261012/` | 以上全部都在備份內 |

重現說明：`step0/runs/REPRODUCE.md`、`step0/runs/EXP2_REPRODUCE.md`。

### 資料授權

- 資料來源 SHIFT15M 為 **CC BY-NC 4.0**（非商業用途）。本 repo **不含任何原始圖片**。
- `step0/manifests/` 內只有 SHIFT15M 的匿名化 item / outfit / user id（hash 或數字 id）與切分清單，不含個人資料；僅供研究重現，須遵守原資料集授權（CC BY-NC 4.0）。
- 官方資料集授權檔與下載腳本在 `step0/docs/official/`（`LICENSE.CC`、`LICENSE.MIT`）。第三方 `shift15m` 套件（`step0/pkg/`）依既有 `step0/.gitignore` 不入版控。

---

## Status (English)

- **Research line paused/closed on 2026-10-13 as a bounded negative result**: no new reliability method under temporal shift on SHIFT15M was found worth adopting. See [`FINAL_RESEARCH_REPORT.md`](FINAL_RESEARCH_REPORT.md).
- This repo is **code + docs only**. Large artefacts (features/data ~42 GB, `*.pt` checkpoints, `main_bootstrap.pkl`, `step0/pylibs/`, one 7 MB near-dup flags file) are **not in git**; they live under `step0/` locally and in the full backup at `/home/chi/backups/outfitmatch-research-20261012/`.
- Dataset: SHIFT15M, **CC BY-NC 4.0**, non-commercial only. No raw images are included; manifests contain only anonymised ids.
- Follow-up product work lives in the separate private repo `theshy7414/outfitmatch-v2` (reuses only the conclusions/architecture, not data or weights).
