# Experiment 2（Prediction Reliability / Error Detection）現況稽核報告

- 稽核時間：2026-10-12 22:15–22:20（UTC+8）。唯讀稽核：沒有訓練任何模型，沒有修改任何既有程式、manifest、checkpoint、預測輸出。
- 唯一新增的檔案就是這份報告。AUROC 的抽查重算是用 `python3` 直接讀 `scores.npz`，沒有寫入任何結果資料夾。
- 讀者：學生本人（繁體中文）與 GPT（沒有 repo 權限）。技術名詞保留英文。所有數字都註明來源檔；來源檔路徑一律相對於 `/home/chi/code/outfitmatch-research/`。

> **先講最重要的三件事**
> 1. **Experiment 2 不只存在，而且已經跑完主體分析**（pre-registration、scores、bootstrap、Gate 1/Gate 2、risk-coverage 圖、linear baseline 全部都在磁碟上，時間戳 19:05–20:06）。
> 2. **結果**：Gate 1 通過（MSP 的 error-detection 能力在 temporal shift 下確實變差）；**Gate 2 兩個候選方法（kNN-pair、ensemble MI）都沒有通過，而且是大幅輸給 MSP**（AUROC 低 0.06–0.11，CI 全部在 0 的負側）。目前唯一「贏 MSP」的是 ensemble-mean-MSP，贏幅小（AUROC +0.02）、risk@80% 沒有顯著改善、成本 5 倍。
> 3. **缺口**：沒有一份整合的最終報告（go/no-go、研究者與 ChatGPT 的分歧）；舊的 Phase 2 報告仍使用 "DECOMPOSITION / excess degradation" 字眼；上次被中斷的 Cursor 對話紀錄**不在磁碟上**（見 §2.1）；稽核期間工作目錄 `NTU-DBMS-2025-Spring` 與 `ntu-push` 的內容被某個外部動作刪除/清空（見 §4.1，請立刻確認）。

---

## 1. 研究目標（用白話說）

**主問題**：Can we detect unreliable outfit-matching predictions under temporal distribution shift？
也就是：模型在「訓練年份之外的年份」做穿搭配對時，我們能不能在**不改模型**的前提下，找出「這題它大概會答錯」的題目？

**三個階段如何連起來**

| 階段 | 做什麼 | 狀態 |
|---|---|---|
| Phase 1 | 在 Task B（FITB：給一套穿搭少一件，從 4 個候選選出缺的那件）上，用 2013 年資料訓練一個 Visual MLP，只用 `val_id` 選超參數（hidden 256, dropout 0.3, lr 3e-4, wd 1e-2）。控制組：query shuffle、candidate-only、random-label。 | 完成（`runs/phase1/`） |
| Phase 2 | 用同一組設定訓練 5 個 seed（`runs/phase2/visual_seed0..4`），在 `test_id / test_ud / 2014 / 2015 / 2017` 量 accuracy、NLL、ECE。得到：accuracy 61.9% → 50.1% → 46.2% → 45.1%，ECE 0.030 → 0.056 → 0.076 → 0.087。 | 完成（`results/exp1_phase2_report.txt`） |
| **Experiment 2** | **不再追求提高 accuracy**，而是問：用凍結的 5 個 checkpoint，哪一種 reliability score（MSP、kNN-pair、ensemble 的 mean-prob / entropy / MI / vote）最能把「答錯的題目」排在前面？ | 主體完成；缺整合報告（見 §2） |

**範圍限制**（來自指派與 `EXPERIMENT2_PREREG.md`）：不訓練新的 compatibility model、不碰 Task A、不動 OutfitMatch v1、不用 shifted-year 的標籤去調任何 k／threshold／normalization。OutfitMatch V2 的 DBMS/Neo4j 產品線與這條研究線**互相獨立**。

**名詞定義（全文一致使用）**
- 錯誤（error）= top-1 預測 ≠ 標籤。**偵測的 positive class = error。**
- 所有 score 都是 **uncertainty（越大越可能錯）**。例如 `msp_single = −max softmax`。
- AUROC = P(u_error > u_correct) + 0.5·P(tie)。AUROC 0.5 = 亂猜；< 0.5 是「方向反了」，不是可以翻轉的小事。
- HCE（high-confidence-error rate）= 用 `val_id` 80% coverage 的 threshold τ 固定後，被接受（score ≤ τ）的題目中的錯誤率。
- S-family = 錯誤指的是 **seed-0 單一模型**的錯誤；E-family = 錯誤指的是 **5-seed ensemble** 的錯誤。兩個 family 絕不混在同一張 AUROC 表。

---

## 2. 目前進度 checklist（逐條對照指派）

圖例：✅ Done｜🟡 Partial｜❌ Not done｜❓ Unverifiable

### 2.1 上次對話在哪裡被切斷

- `~/.cursor/projects/home-chi-code-NTU-DBMS-2025-Spring/agent-transcripts/` 裡**只有一份**對話紀錄：本次稽核的對話（`22b1452d-…`，22:15 開始）。其他 project 資料夾裡也沒有任何提到 OutfitMatch 的 transcript。**所以上次真正做 Experiment 2 的那場對話紀錄不在磁碟上**，我無法直接讀到它說了什麼、使用者被告知什麼。（❓）
- 從檔案時間戳重建（`stat` 結果）：
  - 18:46–18:55：Phase 1/2 訓練與報告。
  - 19:05:48：`EXPERIMENT2_PREREG.md`；19:06:20：`runs/exp2/scores.npz`；19:08–19:09：`61_exp2_main.py` 與主報告、圖；19:19：Gate 報告；
  - 19:34–20:05：`63_linear_baseline.py` 與兩輪 grid；**20:06：`runs/EXP2_REPRODUCE.md`（磁碟上最後一個被寫入的研究檔案）**。
  - 22:01：`~/code/ntu-push` 與 `~/code/outfitmatch-research` 被建立/整理（資料夾 mtime 22:01；`ntu-push` 的 git commit `b382081` "Replace v1 project with OutfitMatch V2 research track" 與 README 都是這時產生）。
- 結論：**研究工作最後停在 20:06（linear baseline 完成 + 重現說明），之後是打包／推送階段（22:01–22:14），並沒有寫整合最終報告。** 切斷點大約就在「打包成 repo」之後、「寫最終報告／交付 GPT」之前。這是由時間戳推論，不是由對話紀錄證實。

### 2.2 逐條對照

| # | 指派要求 | 狀態 | 證據 |
|---|---|---|---|
| 1a | 措辭改成 "descriptive baseline-relative degradation"，不稱因果分解、不加總 | 🟡 | 新檔已改：`EXPERIMENT2_PREREG.md`「Wording constraints」、`scripts/63_linear_baseline.py` 與 `results/exp2_linear_baseline_report.txt` 都寫 "descriptive only: not additive, not a causal decomposition"。**但舊檔沒改**：`scripts/53_phase2_report.py` 第 80–84 行與 `results/exp1_phase2_report.txt` TABLE 3 仍叫 "DECOMPOSITION / benchmark difficulty shift / excess degradation"。 |
| 1b | 補充 linear baseline（只用 train/val_id；報參數量、ID/OOD acc、變化） | ✅（有一個命名疑慮） | `scripts/63_linear_baseline.py`、`results/exp2_linear_baseline_report.txt`、`runs/exp2/linear/seed0..4`。參數量 262,272（MLP 1,245,953）。見 §3.6。 |
| 2a | Gate 1 保留原門檻，註記 2014 ECE +0.026 < +0.03 | ✅ | `EXPERIMENT2_PREREG.md` 第 10 行明寫；`results/exp2_gates_report.txt`（2014 ECE 對 test_ud +0.021、對 test_id +0.025，皆 fail） |
| 2b | 區分 confirmatory 與 exploratory | ✅ | `EXPERIMENT2_PREREG.md`「CONFIRMATORY（C1、C2）／EXPLORATORY」兩節 |
| 2c | user-cluster bootstrap 計算 ECE、error detection、HCE 差異 | ✅（ECE 的 CI 只有 Gate 1 的單模型 MSP） | `62_exp2_gates.py`，B=2000，independent bootstrap；`61_exp2_main.py` 的 paired bootstrap 只做 AUROC/risk/coverage，**ensemble 與 kNN 的 ECE/NLL/Brier 只有點估計**（`exp2_main_report.txt` TABLE 7） |
| 2d | 若 Gate 1 strict 失敗 → 標 exploratory | ✅（不需要） | Gate 1 **通過**（OR 條件，見 §3.1），所以 C1/C2 維持 confirmatory 地位；但 2014 ECE 單項是 fail |
| 3a | 2014/2015 為 primary，2017 auxiliary；test_id/test_ud 分開 | ✅ | PREREG「Provenance」誠實揭露「主要年份是看過 Phase 2 accuracy 之後才選」；所有表都含 5 個 set |
| 3b | 每年報 #instances、#users、effective users | ✅ | `exp2_main_report.txt` 第 0 節表（見 §3.2） |
| 3c | user-cluster CI、user-capped 敏感度 | ✅ | TABLE 9「USER-CAPPED ≤5/user」+ 20 次抽樣穩定性；Gate 1 也做 cap-retention |
| 3d | seen/unseen item 與 clean subgroup | ✅（部分 stratum 樣本太小） | TABLE 9：user seen/unseen、positive id-SEEN、visually novel、novel-and-clean。2015 的 id-seen 只有 103 題、2017 只有 28 題 |
| 3e | 報告全部年份，含不利的 2017 | ✅ | 全部表格皆含 2017 |
| 4A | MSP 單模型 + ensemble-mean MSP | ✅ | `msp_single`、`ens_msp` |
| 4B | kNN（query–predicted-candidate pair embedding；bank 來自 train；k/threshold/normalization 只用 val_id；檢查是否只反映年份） | ✅ | `60_extract_scores.py`（bank = train outfits 的 (query, TRUE candidate) 256-d hidden，L2-normalised，30,816 筆，rng 4242）；k 由 val_id AUROC 選（5 個 seed 全選 k=10）；**沒有任何 per-year normalization**；TABLE 8 檢查 year/density |
| 4C | 5-seed ensemble：mean prob、entropy、MI/disagreement，且對照 ensemble-mean MSP | ✅ | E-family 與 S-family；`ens_entropy`、`ens_mi`、`ens_vote`、`ens_vote_peers` |
| 5a | AUROC（定義 positive=error、方向一致） | ✅ | PREREG「Definitions」＋ `exp2_lib.py` docstring ＋ val_id 方向檢查（0.5 以上） |
| 5b | risk@80% coverage（各 set 自己的排序） | ✅ | TABLE 3 |
| 5c | val_id threshold 固定後轉移到各年的 coverage / risk | ✅ | TABLE 4（含 accepted-AND-wrong 佔全體比例）；與 per-year top-80% 分表呈現 |
| 5d | accuracy、NLL/Brier/ECE、confidence 分布、HCE、runtime/memory | ✅（僅點估計，GPU only） | TABLE 7、TABLE 10、`runs/exp2/cost.json` |
| 5e | 每個方法對 MSP 的 paired user-cluster bootstrap | 🟡 | AUROC（TABLE 2）與 risk@80%（TABLE 3 下方 improvement）有 paired CI。**固定 threshold 的 coverage／risk 只有各方法自己的 CI，沒有「對 MSP 的 paired 差」表。** |
| 6 | 六個詮釋問題的答案 | 🟡 | 數據都在，但**沒有任何文件回答這六題**（只有 `ntu-push/README.md` 一句 headline）。本報告 §3 與 §6 補上 |
| 7 | 最終報告七項（含 go/no-go、與 ChatGPT 最大分歧） | ❌ | 磁碟上沒有整合報告；「研究者最不同意 ChatGPT 之處」只有研究者本人能寫，沒有任何檔案記錄。本報告只能提出稽核者觀點（§6.3） |
| 保存 | 預測輸出、confidence、統計腳本、可重現指令 | ✅ | `runs/phase2/*/preds.npz`、`runs/exp2/scores.npz`、`main_bootstrap.pkl`、`scripts/60–63`、`runs/REPRODUCE.md`、`runs/EXP2_REPRODUCE.md`；manifests 經 `sha256sum -c` 全部 OK |
| 附加（spec §2.1） | 2014 訓練 replication manifest「Gate 1 通過才評估」 | ❌（尚未執行） | 凍結的 `S1_tr2014_*` manifest 存在；但沒有任何 tr2014 的訓練或評估輸出。這會需要訓練新模型，與本輪「不訓練」的約束衝突，所以屬「選擇性、需另行授權」 |

---

## 3. 實際數字（皆讀自結果檔）

### 3.1 Gate 1（confirmatory；單模型 MSP；5 seed 平均）

來源：`results/exp2_gates_report.txt`。門檻不變：ΔECE ≥ +0.03 **或** ΔAUROC(MSP) ≤ −0.03 **或** ΔHCE ≥ +5 pp；且 95% user-cluster CI 排除 0、5/5 seed 同方向、≥3 seed 達門檻、user-cap 後保留 ≥50% 幅度、placebo CI 涵蓋 0；3 個年份中 ≥2 個通過。

Set-level（seed 平均）：

| set | ECE | AUROC(MSP) | HCE % | τ 下 coverage % |
|---|---|---|---|---|
| test_id | 0.031 | 0.703 | 33.0 | 81.0 |
| test_ud | 0.035 | 0.711 | 35.7 | 78.8 |
| test_2014 | 0.056 | 0.654 | 43.8 | 68.7 |
| test_2015 | 0.076 | 0.639 | 47.6 | 64.3 |
| test_2017 | 0.087 | 0.633 | 49.1 | 65.1 |

對 **test_ud（primary reference）** 的差（[95% CI]）：

| 年份 | ΔECE（門檻 +0.03） | ΔAUROC（門檻 −0.03） | ΔHCE（門檻 +5pp） |
|---|---|---|---|
| 2014 | **+0.021 [+0.004,+0.028] → fail**（point < 門檻、僅 2/5 seed ≥ 門檻、cap 保留 40%） | −0.057 [−0.076,−0.039] PASS | +8.1pp [+5.8,+10.6] PASS |
| 2015 | +0.041 [+0.023,+0.047] PASS | −0.072 [−0.092,−0.054] PASS | +11.9pp [+9.5,+14.4] PASS |
| 2017 | +0.052 [+0.031,+0.061] PASS | −0.078 [−0.098,−0.059] PASS | +13.4pp [+10.9,+16.2] PASS |

- 對 test_id（secondary）：2014 ECE +0.025 [+0.010,+0.029] 同樣 fail；其餘 8 項全 PASS。
- Placebo（test_2014 隨機 user 對半）：ΔECE +0.007 [−0.008,+0.021]、ΔAUROC −0.004 [−0.021,+0.013]、ΔHCE +1.2pp [−1.0,+3.6]；200 次隨機切分的 95% 範圍約 ±0.016（AUROC）、±2.4pp（HCE）。
- **Gate 1 結論：兩個 reference 都 PASS（2014、2015、2017 三年都至少有一項指標通過全部條件）。** 2014 的 ECE 單項沒過，但規則是 OR，所以 Gate 1 整體通過；這點要在報告裡明說，不要寫成「ECE 三年都過」。

### 3.2 每年樣本規模與 user 集中度

來源：`results/exp2_main_report.txt` 第 0 節。

| set | n 題 | users | effective users（Kish） | acc seed0 | acc ens | acc 5-seed 平均 |
|---|---|---|---|---|---|---|
| val_id | 3851 | 1380 | 319 | 59.7 | 61.8 | 59.6 |
| test_id | 3172 | 1277 | 344 | 62.3 | 65.1 | 61.9 |
| test_ud | 3851 | 593 | **53** | 59.2 | 60.4 | 58.8 |
| test_2014 | 7702 | 1937 | 453 | 50.3 | 52.3 | 50.1 |
| test_2015 | 7702 | 1361 | 352 | 45.7 | 48.3 | 46.2 |
| test_2017 | 7702 | 373 | **53** | 44.7 | 46.6 | 45.1 |

→ 2017 與 test_ud 的 effective users 都只有約 53，**不能因為 7702 題就說統計力高**。

### 3.3 Error-detection AUROC（S-family，seed-0 錯誤）與對 MSP 的 paired 差

來源：`results/exp2_main_report.txt` TABLE 1、TABLE 2（user-cluster 95% CI；paired = 同一組 bootstrap resample）。

| 方法 | test_id | test_ud | 2014 | 2015 | 2017 |
|---|---|---|---|---|---|
| msp_single | 0.700 | 0.705 | 0.652 [0.639,0.665] | 0.643 [0.629,0.656] | 0.629 [0.616,0.642] |
| knn_pair | 0.573 | 0.580 | 0.545 | 0.552 | 0.530 |
| ens_msp | 0.723 | 0.732 | 0.675 | 0.664 | 0.652 |
| ens_entropy | 0.707 | 0.719 | 0.664 | 0.651 | 0.639 |
| ens_mi | 0.610 | 0.623 | 0.571 | 0.558 | 0.567 |
| ens_vote_peers | 0.688 | 0.687 | 0.647 | 0.639 | 0.636 |
| meta（metadata 控制組） | 0.556 | 0.541 | 0.537 | 0.528 | 0.530 |

對 msp_single 的 paired ΔAUROC（2014 / 2015 / 2017）：
- knn_pair：**−0.107 [−0.123,−0.091] / −0.091 [−0.108,−0.074] / −0.099 [−0.114,−0.084]**
- ens_mi：**−0.081 [−0.099,−0.064] / −0.084 [−0.103,−0.066] / −0.061 [−0.079,−0.044]**
- ens_msp：**+0.023 [+0.015,+0.032] / +0.021 [+0.012,+0.030] / +0.024 [+0.015,+0.032]**
- ens_entropy：+0.012 / +0.008 / +0.011（CI 邊緣）；ens_vote_peers：−0.004 / −0.004 / +0.007（CI 含 0）。

**E-family（錯誤 = ensemble 的錯誤；對 ens_msp 的差）**（TABLE 1/2 [E]）：ens_msp 0.670 / 0.651 / 0.649；ens_mi 0.556 / 0.541 / 0.547（Δ −0.114 / −0.110 / −0.101，CI 皆 <0）；knn_pair Δ −0.141 / −0.102 / −0.127；單模型 MSP Δ −0.035（三年皆 CI<0）。

**抽查重算（我自己用 numpy 從 `scores.npz` 重算，不依賴作者程式碼）**：msp_single AUROC test_2014 = 0.652、test_2017 = 0.629；knn_pair = 0.545 / 0.530；ens_msp(S) = 0.675 / 0.652；E-family ens_mi = 0.556 / 0.547。**與報告完全一致。**

### 3.4 Risk@80% coverage（各 set 自己的排序）與固定 val_id threshold

來源：`exp2_main_report.txt` TABLE 3、TABLE 4、TABLE 5。

**Risk@80%（error rate among the 80% most confident；S-family）**

| | base error | msp | knn_pair | ens_msp | ens_mi |
|---|---|---|---|---|---|
| 2014 | 49.7 | 45.9 | 48.6 | 45.4 | 47.8 |
| 2015 | 54.3 | 50.8 | 53.5 | 50.6 | 53.2 |
| 2017 | 55.3 | 52.1 | 54.7 | 51.7 | 53.5 |

對 MSP 的改善（pp，正 = 更好，paired CI）：knn_pair −2.67 [−3.40,−1.89] / −2.65 / −2.63（三年 CI 皆 <0）；ens_mi −1.92 / −2.43 / −1.43（皆 <0）；**ens_msp +0.46 [−0.11,+1.03] / +0.19 [−0.32,+0.75] / +0.44 [−0.10,+0.92]（CI 含 0，不顯著）**。

**「只留最有信心的預測，錯誤率會下降嗎？」** 會，但幅度有限：以 MSP 在 2014，保留全部 → 49.7%，保留 80% → 45.9%，保留 50% → 39.1%（TABLE 6）。2017：55.3% → 52.1%（80%）→ 46.3%（50%）。即使只留一半題目，2017 仍有 46% 錯誤率。

**固定 val_id threshold（coverage % / risk among accepted %）— S-family**

| 方法 | test_id | test_ud | 2014 | 2015 | 2017 |
|---|---|---|---|---|---|
| msp_single | 80.2 / 32.1 | 79.0 / 35.3 | 68.3 / 43.5 | 63.0 / 47.8 | 62.7 / 49.1 |
| ens_msp | 82.0 / 32.8 | 78.5 / 34.7 | 67.5 / 42.4 | 62.8 / 46.6 | 62.8 / 48.4 |
| knn_pair | 79.7 / 36.2 | 80.6 / 38.8 | 79.9 / 48.6 | 76.0 / 53.0 | 73.2 / 54.5 |
| ens_mi | 80.4 / 34.8 | 79.7 / 37.7 | 80.0 / 47.8 | 79.8 / 53.2 | 78.1 / 53.3 |

詮釋：MSP 的 coverage 會在 shift 下自動掉到 63–68%（它「察覺」輸入變得沒把握），kNN 與 MI 的 coverage 幾乎不動（73–80%）— 它們**沒有察覺 shift**，所以固定 threshold 下 risk 更高。注意固定 threshold 的 risk 受 coverage 影響，不能單看 risk 低就說比較好（`ens_entropy` 的 accepted-and-wrong 佔全體比例最低，27.8–28.7%，但 coverage 也較低）。

### 3.5 kNN 是不是只在偵測「年份／密度」？（exploratory，TABLE 8）

- 平均 kNN 距離隨年份上升：test_id 0.733 → 2014 0.747 → 2015 0.762 → 2017 0.777（seed 0）。density shift 是真的。
- **Year-detection AUROC**（分 test_id vs 該年；0.5 = 分不出）：msp 0.615 / 0.650 / 0.656；**knn 0.532 / 0.576 / 0.615**；ens_mi 0.524 / 0.529 / 0.547。
- Pooled error AUROC（ID+年份合併）與 within-set 對照：msp 0.672 vs 0.652（2014）；knn 0.555 vs 0.545。差距不大，表示「合併」並沒有靠年份偵測偷到太多分。
- kNN 與 MSP 在各 set 內的 Spearman 相關只有 0.14–0.30 → 兩者抓到的不是同一種訊號，但**kNN 的那個訊號跟「答錯」關係很弱**（within-set AUROC 0.53–0.55）。
- 5 個 seed 的 kNN−MSP（點估計）：test_2014 −0.102 ~ −0.119；2015 −0.088 ~ −0.102；2017 −0.082 ~ −0.112 → 5/5 seed 同方向。
- 結論：kNN 的 AUROC 弱，是因為它對「錯誤」本身資訊少，不是因為被年份混淆而被誇大。

### 3.6 子群與 user-cap 敏感度（exploratory，TABLE 9；B=1000）

- **visually novel positive and clean**（n = 6535 / 7547 / 7668 for 2014/15/17）：msp 0.652 / 0.642 / 0.628；knn Δ −0.117 / −0.093 / −0.098；ens_mi(E) Δ −0.112 / −0.112 / −0.101。**與全體幾乎相同**（因為 2015/2017 幾乎所有題目的 positive 都是 visually novel）。
- **user seen / unseen**：兩組 Δ 都負（2014：seen −0.117、unseen −0.104）。
- **positive id-SEEN（item 在 train 出現過）**：2014（n=979）knn Δ −0.044 [−0.091,+0.006]（唯一 CI 觸及 0 的）；2015（n=103，**樣本太小不可解讀**）；2017 n=28 不報。
- **User-capped ≤5/user**：msp 0.645 / 0.651 / 0.633；knn Δ −0.087 / −0.096 / −0.111（CI 全 <0）；20 次抽樣下 2017 的 msp AUROC 範圍 0.597–0.636，**非常寬**（user 太集中）。
- 結論方向不依賴 user-cap；但 2017 的任何細節差異都不該被解讀。

### 3.7 Calibration、confidence 分布（TABLE 7，點估計，無 CI）

| set | seed0 acc | seed0 ECE | seed0 conf−acc | ensemble ECE | ensemble conf−acc | seed0 NLL → ens NLL |
|---|---|---|---|---|---|---|
| test_id | 62.3 | 0.036 | +1.1pp | 0.049 | **−4.9pp** | 0.900 → 0.857 |
| test_2014 | 50.3 | 0.060 | +6.0pp | 0.017 | +0.6pp | 1.137 → 1.080 |
| test_2015 | 45.7 | 0.085 | +8.5pp | 0.026 | +2.6pp | 1.219 → 1.156 |
| test_2017 | 44.7 | 0.091 | +9.1pp | 0.041 | +4.1pp | 1.252 → 1.179 |

- ensemble 在 shifted 年份的 ECE 明顯比單模型小（2014：0.017 vs 0.060），但在 test_id 反而 under-confident（−4.9pp，ECE 0.049 比單模型 0.036 差）。→ ensemble 的「校準改善」是方向性的平移，不是單調改善；**沒有 CI，不要寫成已證實的結論。**
- 高信心（max-p ≥ 0.8）的題目占比：test_id 約 21.8% → 2014 10.2% → 2015 7.7% → 2017 7.2%（seed 0）。

### 3.8 運算與記憶體成本（TABLE 10、`runs/exp2/cost.json`；RTX 3070 Ti，test_2015 7,702 題，features 已在 GPU）

| | 時間 | 倍率 | 參數／額外儲存 | 額外訓練 |
|---|---|---|---|---|
| 單模型 MSP | 42.7 ms（5.55 µs/題） | 1× | 1,245,953 參數 | 無（訓練 20.8 s） |
| ensemble（5 模型） | 217.5 ms | 5.1× | 6,229,765 參數（5×） | 訓練 90.8 s（5×，已存在） |
| kNN-pair | 60.3 ms（含取 hidden + 搜尋） | 1.4× | bank 31.6 MB；建 bank 183 ms | 無 |

限制：只量 GPU；暴力內積搜尋，沒用 FAISS；`knn_peak_gpu_MB_incl_data = 3641 MB` **包含整份 feature 資料，不是 kNN 的增量記憶體**；沒有 CPU／serving 延遲。

### 3.9 Linear baseline 補充（`results/exp2_linear_baseline_report.txt`）

- 模型：4096→h 線性 encoder（無非線性）、query mean-pool，分數 = ⟨c,e⟩ + w·e。**注意：這其實是 bilinear（對輸入是二次式）的 low-rank 模型，不是嚴格的 linear classifier**；報告與口頭描述請寫 "linear-encoder bilinear baseline"。
- 只用 train 訓練、val_id NLL 選超參數（24 組 grid，seed 0）；選出 hidden 64、dropout 0.1、lr 1e-4、wd 0.1；參數量 262,272。
- 準確率（5 seed 平均 %）：

| set | cosine | linear | MLP |
|---|---|---|---|
| test_id | 34.3 | 53.6 | 61.9 |
| test_ud | 32.4 | 51.5 | 58.8 |
| 2014 | 29.3 | 43.0 | 50.1 |
| 2015 | 28.6 | 40.7 | 46.2 |
| 2017 | 28.5 | 39.8 | 45.1 |

- 相對 test_id 的 accuracy 變化（pp）：2014：cosine −5.0、linear −10.6、MLP −11.8（MLP−linear = −1.2 [−2.6,+0.2]）；2015：−5.7 / −12.9 / −15.8（−2.9 [−4.3,−1.5]）；2017：−5.8 / −13.8 / −16.9（−3.1 [−4.6,−1.5]）。
- **揭露**：第一輪 4 組 grid（hidden 256，lr 3e-4/1e-3）選到 lr 邊界、best_epoch 1–3，accuracy 只有 45.1%（`exp2_linear_baseline_report_firstgrid.txt`）；之後在 val_id 上擴大 grid（沒用 shifted 資料）。**兩輪的「MLP 掉幅 − linear 掉幅」差很多（2014：−4.4 vs −1.2）**。這代表這個差值**高度依賴 baseline 強弱**，正好支持「不要把它當因果拆解」的措辭，也代表舊報告的 "excess degradation（−6.9 pp，相對 cosine）" 不宜再引用。

---

## 4. 問題與風險

### 4.1 緊急（環境／備份）
1. **稽核進行中，`/home/chi/code/NTU-DBMS-2025-Spring/`（本任務指定的工作區）與 `/home/chi/code/ai` 消失，`/home/chi/code/ntu-push/` 內容被清到只剩空的 `.git/objects`**（22:17–22:18 間，我用 `ls` 前後比對觀察到；我只做唯讀指令，不是我造成的）。稽核開始時（22:15）`ntu-push` 還有完整檔案與 commit `b382081`。請立刻確認這是不是你自己（或某個 Cursor／git 指令）做的。
2. 因此本報告**無法存在指定位置** `/home/chi/code/NTU-DBMS-2025-Spring/EXPERIMENT2_STATUS_REPORT.md`（該資料夾已不存在，我不想在別人整理到一半的路徑上重建它），改存於 **`/home/chi/code/outfitmatch-research/EXPERIMENT2_STATUS_REPORT.md`**。
3. 即使 `ntu-push` 的 git 還在，**大檔都不在 git 裡**（`.gitignore`：`step0/data/`、`*.pt`、`*.npy`、`*.pkl`、`pylibs`）。目前**唯一一份** VGG16 features、5 個 seed 的 checkpoint（`runs/phase2/visual_seed0..4/best.pt`）、`main_bootstrap.pkl` 都只在 `outfitmatch-research/step0/` 底下。**請先備份這個資料夾（features 約 3.2 GB required 子集 + 19 GB 全集），不要刪。**
4. 無法確認 `ntu-push` 有沒有成功 push 到遠端（我沒來得及讀 `git remote`，之後 `.git/config` 就被清掉了）。

### 4.2 實驗設計／程式檢查結果（已讀程式碼）
- ✅ **Score 方向與 positive class 一致**：`exp2_lib.py`、`exp2_data.py` 都以 error=1、u 越大越不確定；val_id AUROC 全部 > 0.5（msp 0.704、knn 0.583、ens_mi 0.624、meta 0.515）。
- ✅ **沒有 shifted-year 標籤進入 k/threshold**：k 只用 `val_id`（`choose_k`）；τ 只用 `val_id` 80% quantile（`tau_val`、`tau_seed`）；沒有任何 normalization。
- ✅ **bank 沒有汙染**：`data.sample_train` 只用 train outfits（`S1_tr2013_outfits` 的 `train` role，且特徵標準化 mean/std 也只用 train items）；val_id／test 不在 bank 內。
- ✅ **Ensemble 用對 seed**：`runs/phase2/visual_seed0..4`，`60_extract_scores.py` 逐一 assert `cfg["variant"]=="visual"` 並檢查重算 logits 與存檔 `preds.npz` 差 < 2e-3（log：「logits match stored preds」）。
- ✅ **Bootstrap 是 user-cluster**（`boot_weights`：以 user 為單位重抽，同 user 的所有題目一起）；同一 set 的各方法使用同一組 resample（paired）；兩個不同 set 用獨立 resample。
- ✅ **Checkpoint 沒被改動**：`best.pt` mtime 18:51–18:54（早於 Exp2 所有輸出）；manifest 的 `sha256sum -c` 全 OK。
- 🟡 **Pre-registration 時間證據很薄**：PREREG 19:05:48，scores 19:06:20，`exp2_data.py`／`61_exp2_main.py` 19:07–19:08。順序是對的，但沒有 git commit 或外部 hash 證明 PREREG 之後沒被改過。→ 建議現在就記錄 sha256（PREREG：`a9b087e0…`；scores.npz：`43b8c186…`；gates 報告：`7f23777c…`；main 報告：`6eb14a63…`）。
- 🟡 **ECE 數字兩處不一致**：Phase 2 報告 test_id/ud ECE = 0.030/0.036，Gate 報告 = 0.031/0.035（`exp2_lib.Ece` 的 bin 指派略不同）。所以「2014 ECE 相對 test_id」是 +0.026（Phase 2）還是 +0.025（Gate）取決於版本；兩者都 <0.03，結論不變，但文件要註明用哪個。
- 🟡 **`ens_vote_peers`、`ens_vote` 分數極為離散**（只有 5 或 6 個值），固定 threshold 時 coverage 變成 83.1% / 91.5%，不是 80%；不能與其他方法的固定 threshold 表直接比。
- 🟡 **kNN 的 k=10 是候選集的下界**（5 個 seed 全選 10）；沒有試 k<10。但因差距達 0.1 AUROC，不影響結論。
- 🟡 **ensemble 只有 5 個同設定模型**（同架構、同資料、不同 init/順序），多樣性低 → MI 估計雜訊大。這是「此設計下 MI 輸給 MSP」，不是「所有 ensemble disagreement 都沒用」。
- 🟡 `scripts/52_phase2_train.py` docstring 指向 `results/exp1_phase2_train_log.txt`，但該檔不存在（訓練輸出可能沒 tee）。`run_info.json`／`log.jsonl` 仍在，影響可重現性不大。
- ❓ 舊對話中使用者被告知什麼、哪些結論已經轉述給 GPT，無法驗證。

### 4.3 與先前說法的一致性
- 先前已知數字（Visual MLP 61.9/50.1/46.2/45.1；cosine 34.3/28.5；ECE 0.030→0.056→0.076→0.087；conf−acc +0.7→+5.5→+7.5→+8.6）與 `exp1_phase2_report.txt` 一致。
- 先前的 "excess degradation" 論述被 linear baseline 弱化（見 §3.9）。
- `ntu-push/README.md` 的 headline（「MSP 的 AUROC 掉 0.06–0.08，Gate 1 通過；kNN 與 MI 沒贏過 MSP」）與 `exp2_gates_report.txt` 一致。

---

## 5. 與要求不一致或可能被誤解的地方（claims 可能超出證據）

1. **「Gate 1 通過」≠「temporal shift 導致 calibration/可靠度變差」。** 它只證明：這些年份的 test set 對這個固定模型而言，MSP 偵錯誤能力下降、ECE 上升。Placebo 只排除了「隨機 user 對半」；不能排除 user 組成、item novelty（ID 的 positive 約 71% 在 train 出現過，2014 僅約 13%，2015/2017 約 1%）、類別組合等與年份混在一起的因素。matched-novel 分析（Phase 2 TABLE 5）顯示掉幅縮小但仍在（MLP 2014 −9.5pp），所以不是「全部來自 item novelty」，但也不是因果證據。
2. **HCE 通過 Gate 1 很大程度是機械性的。** HCE 是固定 τ 下被接受題目的錯誤率；accuracy 從 62% 掉到 45–50% 時，這個數字必然上升。真正能代表「error detection 變差」的是 AUROC（三年皆 −0.057 ~ −0.078），ECE 則 2014 沒過。建議報告把 HCE 標成「accuracy-driven descriptive」。
3. **2014 ECE +0.026（或 +0.025）< +0.03 這件事必須原樣呈現**，不能因為 Gate 1 整體 PASS 就省略；2014 的通過是靠 AUROC 與 HCE。
4. **「MSP is sufficient」要小心**：MSP 的 AUROC 只有 0.63–0.65，偵測力是「中等偏弱」。正確說法是「在我們測的方法裡沒有任何一個比 MSP 好（ensemble-mean-MSP 除外，小幅更好）」，而不是「MSP 已經夠用」。
5. **ens_msp 比 msp_single 好（+0.02 AUROC，CI > 0）有一半是因為 ensemble 本身比較準**（accuracy 高 1–3 pp）、且它同時使用了 5 個模型；risk@80% 沒有顯著改善（CI 含 0）。不宜寫成「ensemble 讓不確定度更可靠」，更不能把 calibration 的改善歸功於 disagreement（本實驗已用 E-family 對 ens_msp 做對照，結果是 MI 比 ens_msp 低 0.10–0.11 AUROC）。
6. **kNN 的負面結果只適用於這一個設計**（256-d scorer hidden、query–predicted-candidate pair、train 正例 bank、k=10、brute-force Euclidean）。不能延伸成「density/novelty 方法對穿搭沒用」。bank 的 embedding 也是模型自己訓練過的 in-sample 表示。
7. **2017 結果脆弱**：只有 373 users／effective 53；user-cap 後 msp AUROC 範圍 0.597–0.636。「2017 最差」的敘述可以給，細節比較不要。
8. **test_ud 作為 Gate 1 的 primary reference 也只有 effective users 53**（593 users）。reference 本身 CI 寬，ΔECE 的 CI（如 2014 [+0.004,+0.028]）因此偏寬。
9. **Primary year 的選擇是看過 Phase 2 結果後才做的**（PREREG 已揭露）。Gate 的判斷其實對 3 個年份都有算，所以影響不大，但最終報告要重述。
10. **"linear baseline" 這個名字**：實際是 bilinear low-rank；且掉幅差對 baseline 強度敏感（§3.9），不能當成「任務本身難度」的估計。
11. **ECE/NLL/Brier 的 ensemble 與 kNN 數字沒有 CI**，ensemble 的 calibration 結論只能叫 exploratory。
12. **成本結論限於 GPU、小資料（7.7k 題、31k bank）。** 在 DBMS 產品的規模上（大 bank、CPU serving）kNN 的成本沒有量過。

---

## 6. 建議下一步（最小、有界）

### 6.1 補齊 Experiment 2（皆可只讀既有 `scores.npz`／`preds.npz`，**不需訓練**）
1. **先備份、記 hash**：把 `outfitmatch-research/` 整份（至少 `step0/runs`、`step0/results`、`step0/manifests`、`step0/scripts`、`EXPERIMENT*.md`、features、`best.pt`）複製到另一個位置；把 §4.2 的 sha256 寫進最終報告。
2. **寫一份整合最終報告**（目前缺的第 7 項），包含：Gate 1 正式結果（含「2014 ECE 單項 fail」）、MSP/kNN/ensemble 分年比較、paired CI、risk-coverage 圖（`runs/exp2/figures/risk_coverage.png`、`fixed_threshold.png` 已存在）、固定 threshold 表、seen/unseen 與 user-cap、成本表、go/no-go。
3. **補三個小缺口**（純分析，幾十行 numpy）：(a) 固定 threshold 的 coverage／risk **對 MSP 的 paired user-cluster CI**；(b) ensemble 與 kNN 的 ECE／NLL／Brier **CI**；(c) `ens_msp` vs `msp_single` 在 seeds 1–4 的一致性（目前只有 seed 0 的 S-family）。
4. **措辭更正**：在最終報告加一段 erratum，聲明 Phase 2 報告 TABLE 3 的 "DECOMPOSITION / benchmark difficulty / excess degradation" 一律改稱 "descriptive baseline-relative degradation"；不必改舊檔（它們是歷史紀錄），除非你決定要。
5. **linear baseline**：已完成，不需再跑；只需改名為 "linear-encoder bilinear baseline"，並同時列出兩輪 grid 的結果（第一輪的 `firstgrid` 已保留）。
6. tr2014 replication（spec 因 Gate 1 通過而「允許」）：**需要訓練新模型**，與本輪約束衝突，除非你與 GPT 明確同意，否則**不做**。

### 6.2 誠實的 verdict：值不值得繼續？
- **做為一個有界的研究：值得「收尾」。** 你已經有一個乾淨、有 pre-registration、有 user-cluster CI、有 placebo 與 user-cap 的結果：現象（MSP 在 shift 下偵錯變差）成立，而兩個常見的「加強版 reliability 訊號」（kNN-pair、ensemble MI）**都沒有比最簡單的 MSP 好**，且差距遠超過 CI。這是一個可信的 negative result，pre-registered Gate 2 本來就把它定義為「MSP is sufficient / signals do not add」。
- **不建議繼續擴大 reliability 方法的搜尋**（再加更多 kNN 變體、更多 score）——證據顯示信號量太低（AUROC 約 0.53–0.57 vs 0.63–0.65），再花時間邊際效益低，且很容易變成對 shifted-year 的 p-hacking。
- **go/no-go（對下一個「方法開發」階段）：No-go。** 對「寫成報告／報告給課堂」：**Go**（約 1–2 天）。若還想做新問題，較有價值且仍有界的是：「為什麼 AUROC 會掉」（confidence 分布 vs accuracy 的耦合）或簡單的 post-hoc recalibration 對照，且必須事先 pre-register，並維持 val_id-only 調參。
- OutfitMatch V2 的 DBMS/Neo4j 產品線可以直接獨立進行；這條研究線唯一能給產品線的結論是：「若要顯示信心，先用 MSP，需要時用 5-model ensemble-mean 當作小幅改善，不要為 kNN／MI 付額外成本。」

### 6.3 稽核者觀點的「最可能與 ChatGPT 分歧之處」（第 7 項要求的是**研究者本人**的意見，這裡只是候選，供你參考）
1. Gate 1「通過」被當成強結論，但 HCE 是機械性的、2014 的 ECE 沒過；實際乾淨的證據只有 AUROC 下降。
2. 「baseline-relative degradation」＋linear baseline 並沒有解決拆解問題：結果對 baseline 的強度很敏感（−4.4 vs −1.2），建議乾脆不報拆解數字。
3. Primary year（2014/2015）是看過 accuracy 後挑的；2017 effective users 僅 53。
4. Gate 門檻本來就寫「都是我的提議」（`EXPERIMENT1_SPEC.md` §6），不是外部標準。
5. 不同意的重點本身請由你自己寫，因為沒有任何檔案記錄。

---

## 7. 給 GPT 的摘要

1. 專案：OutfitMatch V2 研究線（SHIFT15M，Task B = 4 選 1 的 item FITB，train 2013）。Experiment 2 = 在凍結的 5-seed Visual MLP 上做 error detection（positive = error，score = uncertainty）。
2. 磁碟上 Experiment 2 **已跑完主體**：prereg、scores、user-cluster bootstrap（B=2000, paired）、Gate 1/2、risk-coverage 圖、linear baseline；缺的是整合最終報告與 go/no-go；上次對話紀錄不在磁碟上。
3. Gate 1 **PASS**（對 test_ud 與 test_id 都過，3 年皆過）：ΔAUROC(MSP) −0.057/−0.072/−0.078（2014/15/17，CI 皆<0）；ΔHCE +8.1/+11.9/+13.4 pp；**2014 的 ΔECE 單項 fail（+0.021 對 ud，+0.025 對 id）**，靠 OR 規則過關。HCE 的上升大半是 accuracy 下降的機械結果。
4. Gate 2 **NOT PASSED，兩個候選都 0/3 年**：kNN-pair vs MSP ΔAUROC −0.107/−0.091/−0.099；ens-MI vs ens-MSP −0.114/−0.110/−0.101（paired CI 全 <0）；risk@80% 也比 MSP 差 1.4–3.5 pp。
5. 唯一贏過 MSP 的是 ensemble-mean-MSP：ΔAUROC +0.021~+0.024（CI>0），但 risk@80% 改善 +0.2~+0.5 pp（CI 含 0），成本 5.1× 推論、5× 參數；在 E-family 中，ens-MSP 比單模型 MSP 好 0.035 AUROC（部分因 ensemble 本身更準）。
6. kNN 不是被年份混淆：within-set AUROC 只有 0.53–0.55；kNN／MI 的固定 val_id threshold coverage 幾乎不隨 shift 下降（73–80%），而 MSP 掉到 63–68%。
7. Seen/unseen、novel-and-clean、user-cap（≤5/user）都不改變方向；2015/2017 的 item-seen stratum 樣本太小（103/28）；2017 與 test_ud 的 effective users 都只有約 53。
8. Linear baseline 已完成（262,272 參數；ID 53.6%、2014/15/17 = 43.0/40.7/39.8%）：MLP 與 linear 的掉幅差在 2014 只有 −1.2 [−2.6,+0.2]，且對 baseline 強度很敏感（第一輪較弱 grid 為 −4.4）→ 請不要再用 "excess degradation" 或任何因果拆解，只稱 descriptive baseline-relative degradation。
9. 風險：舊 Phase 2 報告仍用 "DECOMPOSITION"；ECE 在兩份報告差 0.001（0.030 vs 0.031）；prereg 只有檔案時間戳（19:05:48 對 scores 19:06:20）；ensemble/kNN 的 ECE/NLL 沒有 CI；固定 threshold 缺「對 MSP 的 paired CI」。
10. 嚴禁的過度宣稱：temporal shift 導致 calibration 退化（只能說 descriptive）；模型「知道自己的錯」（Gate 2 未過）；ensemble disagreement 改善可靠度（已用 ens-MSP 對照，MI 反而更差）；kNN 對穿搭無用（只限本設計）。
11. 建議：不訓練新模型；補 3 個純分析缺口（固定 threshold paired CI、ensemble/kNN 校準 CI、seeds 1–4 一致性）後寫最終報告；結論是 "MSP is the best of the tested cheap signals; ensemble-mean gives small AUROC gain at 5× cost"。
12. Go/no-go：對「繼續開發更多 reliability 方法」= No-go；對「收尾成 negative-result 報告」= Go；DBMS/Neo4j 產品線完全獨立。
13. 工作區稽核時被外部刪除（`NTU-DBMS-2025-Spring`、`ntu-push` 內容消失）；所有大檔與 checkpoint 只存在於 `outfitmatch-research/step0/`，請務必備份。

---

## 附錄：關鍵檔案路徑（皆在 `/home/chi/code/outfitmatch-research/` 之下）

| 類別 | 路徑 |
|---|---|
| 規格／預註冊 | `EXPERIMENT1_SPEC.md`（§6 Gate 定義）、`EXPERIMENT2_PREREG.md` |
| Frozen manifests | `step0/manifests/`（`MANIFEST_SHA256.txt`，已驗證全 OK） |
| Frozen checkpoints | `step0/runs/phase2/visual_seed0..4/best.pt`（及 `preds.npz`） |
| Exp2 腳本 | `step0/scripts/60_extract_scores.py`、`61_exp2_main.py`、`62_exp2_gates.py`、`63_linear_baseline.py`、`exp2_data.py`、`exp2_lib.py`、`taskb_lib.py`、`run63.sh` |
| Exp2 分數與統計 | `step0/runs/exp2/scores.npz`、`main_bootstrap.pkl`、`cost.json`、`figures/risk_coverage.png`、`figures/fixed_threshold.png` |
| Exp2 報告 | `step0/results/exp2_main_report.txt`、`exp2_gates_report.txt`、`exp2_extract_log.txt`、`exp2_linear_baseline_report.txt`、`exp2_linear_baseline_report_firstgrid.txt` |
| Linear baseline | `step0/runs/exp2/linear/`（28 個 grid 資料夾＝第一輪 4 組 + 第二輪 24 組，另有 `seed0..4`、`selected_config.json`） |
| Phase 1/2 報告 | `step0/results/exp1_phase2_report.txt`、`step0/runs/phase1/`、`step0/runs/phase2/` |
| 重現指令 | `step0/runs/REPRODUCE.md`、`step0/runs/EXP2_REPRODUCE.md` |
| 本報告 | `EXPERIMENT2_STATUS_REPORT.md`（本檔；指定路徑 `NTU-DBMS-2025-Spring/` 已不存在） |
