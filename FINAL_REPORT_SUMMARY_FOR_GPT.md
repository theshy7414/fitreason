# OutfitMatch 研究線最終摘要（可直接貼給 ChatGPT；完整版：FINAL_RESEARCH_REPORT.md）
1. 問題：SHIFT15M、Task B（4 選 1 item FITB，train=2013，凍結 5-seed Visual MLP）。能否在 temporal shift 下偵測「不可靠預測」？不是要提高 accuracy；不訓練新模型。偵測的 positive class = error，所有 score = uncertainty（越大越可能錯）。
2. 資料：test_id 3,172、test_ud 3,851（593 users）、2014/2015/2017 各 7,702。**test_ud 與 2017 的 effective users 只有約 53**；positive item 在 train 見過的比例：ID ~71%、2014 12.7%、2015 1.3%、2017 0.4%（item turnover 與年份糾纏）。
3. Task B 漏洞檢查：candidate-only 25.0、query shuffle 24.7、random-label ≈25（皆 ≈ chance）；metadata 多接近 chance，price-near-context 27–30%；train 共現通道只存在於 ID（cooc 29% → shifted 25%）。官方式 Task A 被 shortcut 主導（category/size 就 60–76%），所以只當參考、沒訓練。
4. 跨年份退化（descriptive，非因果）：accuracy 61.9 → 50.1/46.2/45.1%；ECE 0.030 → 0.056/0.076/0.087；conf−acc +0.7 → +5.5/+7.6/+8.6 pp；MSP AUROC 0.703 → 0.654/0.639/0.633。
5. 措辭：只說 "descriptive baseline-relative degradation"。cosine/co-occurrence 的掉幅不是可加的機制、也不是任務難度。MLP−linear 掉幅差對 baseline 強弱敏感（2014：弱 grid −4.4 vs 最終 −1.2 [−2.6,+0.2]）；舊 Phase 2 報告的 "DECOMPOSITION / excess degradation" 已被取代。
6. Gate 1（pre-registered，門檻未改，OR 規則）PASS，三年皆過：ΔAUROC −0.057/−0.072/−0.078，ΔHCE +8.1/+11.9/+13.4 pp。**2014 的 ΔECE 單項 fail**（+0.021 vs test_ud，0/5 seed 達標；vs test_id +0.025）。HCE 上升大半是 accuracy 下降的機械結果。主要年份 2014/2015 是看過 Phase 2 accuracy 後才選；預註冊只有檔案時間戳為證。
7. Gate 2 **NOT PASSED（0/3 年）**：kNN-pair vs MSP ΔAUROC −0.107/−0.091/−0.099；ens-MI vs ens-MSP −0.114/−0.110/−0.101（paired CI 皆 <0）；risk@80% 也差 1.4–3.5 pp。kNN 與 MI 都幾乎只比 metadata-only control 好 0–0.02。
8. 唯一贏 MSP 的是 ensemble-mean-MSP：ΔAUROC +0.02（CI>0）、risk@80% 差異不顯著、5.1× 推論成本、5× 參數。固定 val_id threshold 下，MSP 的 coverage 隨 shift 掉到 63–68%，kNN/MI 幾乎不動（73–80%）＝沒察覺 shift。
9. kNN 不是被年份混淆（within-set AUROC 只有 0.53–0.55）；結果對 user-cap、seen/unseen、novel-and-clean 子群、5 個 seed 方向一致。但只適用於這個特定 kNN/MI 設計（k=10、5 個同設定模型），不可推廣成「density/ensemble 方法無用」。
10. 限制：2017/test_ud 的 effective users≈53；item-seen 子群 2015 n=103、2017 n=28 不可解讀；tr2014 replication 沒跑（要重訓）；固定 threshold 缺 paired CI；kNN/ensemble 的 ECE/NLL/Brier 缺 CI；沒做 recalibration；GPU-only 成本。
11. 結論（原文）：**我們觀察到跨年份可靠性退化，但尚未找到值得採用的新可靠性方法。** 不主張新穎性、不主張 temporal shift → calibration 的因果機制、不主張模型能辨識自己的錯誤。
12. Go/No-Go：繼續開發更多 reliability 方法 = No-go；收尾成 bounded negative result = Go。可選且「未規劃」：補 paired CI / calibration CI（純分析）、val_id temperature scaling 對照（需先預註冊）、tr2014 replication（需重訓）。
13. 帶入產品（OutfitMatch V2）：若顯示信心，用 MSP 或 ensemble-mean-MSP 當粗略指標並附 caveat（AUROC 僅 0.63–0.70）；不要為 kNN/MI 付成本；SHIFT15M（CC BY-NC、無影像、匿名）不能當產品 demo 資料；可重用 manifest/hash/稽核 harness。DBMS/Neo4j 產品線與研究線獨立。
14. 文件稽核發現：status report 把 2014 ECE 的 per-seed 達標數寫成 2/5（對 test_ud 實為 0/5；2/5 是對 test_id）；ECE 兩份報告差 0.001（binning 實作不同）；`ens_vote_peers` 的 risk@80% 有小幅顯著改善（+0.7~0.8 pp，exploratory）先前未提；fixed_threshold.png 混用 S/E family；spec 與實作的 epoch/patience 設定不一致。無改變結論方向的錯誤。
15. 想請 ChatGPT 協助審視：(a) 結論措辭是否仍有過度宣稱；(b) 「bounded negative result」收尾與「No-go」是否合理；(c) 哪些限制最該在課堂報告優先講。
16. 【研究者最不同意 ChatGPT 之處】：＿＿＿＿＿＿（placeholder，請由研究者本人填寫；本檔不代為決定）
