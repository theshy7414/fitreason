# 03 — Roadmap

假設：一個人、每週約 10–15 小時；總計約 **5–6 週**（~60–80 小時）。每個里程碑結束都要「可 demo / 可測」，並 git tag。

| M | 名稱 | 工時 | 交付物 | 驗收標準（全部要過） |
|---|---|---|---|---|
| **M0** | Scaffold + 兩個 DB + health/evidence | 6–8 h | repo 骨架、`docker-compose.yml`（Neo4j + PG）、tiny seed、FastAPI `/health` `/evidence` | `04_FIRST_TASK.md` 的驗收測試全過；`make up && make test` 從乾淨 clone 可重現 |
| **M1** | 資料管線 + 授權 manifest | 10–12 h | 自拍/Commons 照片入庫（60–150 件）；Fashionpedia 授權過濾腳本與 manifest；Neo4j / PG loader；`/items` `/licenses` | 每個 item 皆有 `license/source_url/attribution`（DB constraint 擋空值）；過濾報告列出保留/排除數（Unsplash/Pexels/NC/ND 各多少）；`items` API 回傳 ≥ 60 件 |
| **M2** | ML：embedding + compat head | 12–16 h | `ml/encoder.py`、`train.py`、`MODEL_CARD.md`、head 權重（≤ 5 MB）、離線 `materialize.py` | FITB（K=4）accuracy **顯著高於** 25% 隨機、category-prior、CLIP-cosine 三個基線（以 3 個 seed 的 mean ± std 報告；差距 > 2 std 才算過）；split 無 image 洩漏（測試有自動檢查）；model card 寫明資料授權與「outfit = 同照片共現」的限制 |
| **M3** | FastAPI 推薦 + evidence | 8–10 h | `/recommend` `/score` `/evidence` `/outfits` `/items` POST | pytest 覆蓋 ≥ 70%（API 層）；`/recommend` p95 < 300 ms（本機 GPU 或 CPU，≤ 1k items）；上傳圖 → 推薦端到端可用；`ood_warning` 對「非衣物圖」測試案例為 true |
| **M4** | React UI | 12–15 h | Upload / Browse / Recommend / Evidence / Licenses 畫面 | 手動 demo 腳本 8 步可跑完（上傳→選 2 件→推薦鞋子→看 evidence）；`npm run build` 無錯誤；Playwright 一個 smoke test；Licenses 頁列出所有圖片來源 |
| **M5** | PostgreSQL 對照 + bench | 8–10 h | `db/postgres/`、`pg_store.py`、`bench/run_bench.py`、`bench/RESULTS.md` | 同一組查詢在兩個 store 的**結果集合相同**（測試檢查）；1k/10k/100k（合成）三個規模的 p50/p95 表；結論段如實反映數據（含 Neo4j 沒贏的情況） |
| **M6** | 收尾與作品集化 | 6–8 h | README 截圖/GIF、demo 影片腳本、`docker compose up` 一鍵、CI（GitHub Actions：lint + tests）、備份 | 乾淨機器 clone 後 ≤ 10 分鐘跑起來；CI 綠；README 的所有指令實測過；依 `01_RECOVERY_AND_BACKUP.md` 3.1 推到 private repo（公開前再逐項確認授權） |

## 依賴順序

M0 → M1 → M2 → M3 → M4；M5 可在 M3 後與 M4 並行。M2 的訓練不阻塞 M3：M3 先用 CLIP cosine 當 placeholder scorer（同一個 `CompatScorer` 介面）。

## 明確邊界（避免研究線無限擴張）

1. **研究線已結案**：`/home/chi/outfitmatch/` 的研究線只讀、不再新增實驗；`v2/` 不 import 它、不複製其 data/features。
2. V2 允許的「研究性」工作只有 M2 的**一次**訓練與評估（基線三個 + head 三個 seed）。超參數搜尋上限：**最多 12 組設定、一個下午**；超過就凍結目前最佳。
3. **不做**：新的 reliability/uncertainty 方法、temporal shift 分析、Task A、ensemble 校準；`ensemble=true` 只能在 M6 後作為選配，且須有明確使用案例。
4. 任何「再多做一個實驗會更好」的想法 → 寫進 `docs/IDEAS.md`，不當週開工。
5. 模型成績**不是**驗收主角；驗收的是「勝過基線、可重現、授權乾淨、能解釋」。若 M2 結果不勝過 CLIP-cosine 基線，**如實寫在 model card，產品上線用 CLIP-cosine + 圖共現混合排名**，不再延長 M2。
6. 時間盒：每個里程碑超過預估 **1.5 倍**就停下來縮 scope（砍功能，不砍驗收）。
7. 資料授權有疑義的圖 → 直接排除，不糾結。

## 風險與備案

| 風險 | 備案 |
|---|---|
| Fashionpedia 過濾後張數太少 | 加入自拍照片自建 outfit（用自己衣櫃拍 30–50 套）；或降低為 pairwise 訓練 |
| 同照片共現當 outfit 品質不佳 | 在 UI/README 誠實標示；排名混合 CLIP-cosine 與 co-occurrence |
| Node 未安裝 | M4 前裝 `fnm`/`nvm`（見 04） |
| Neo4j vector index 版本問題 | 固定 image `neo4j:5.26`，CI 同版本 |
| 上傳圖隱私 | 預設只存本機 volume；去 EXIF；Licenses 頁註明 demo 圖來源 |
