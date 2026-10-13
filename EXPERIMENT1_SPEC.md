# Experiment 1 — Specification v1 (frozen with manifests `v1`)

Title: *Detecting Unreliable Fashion Matching Predictions under Temporal Distribution Shift*
Data: SHIFT15M (IQON), metadata + frozen VGG16 fc6 features. No raw images. CC BY-NC 4.0: derived data stays local.
Goal: find reproducible failure phenomena and their sources. NOT to maximise accuracy. No new model/loss.

Reproduce everything: `step0/scripts/20_build_manifests.py` -> `21_audit_and_baselines.py` (hashes in `step0/manifests/MANIFEST_SHA256.txt`).

## 1. Tasks

### Task A — official-style set-to-set matching (reference task)
* Pool: official filter (>=4 distinct category_id1 AND like_num >= 50), file order, NO dedupe (as official).
* Split (official procedure, seed 0; same algorithm, pandas `sample(frac=1)` + sklearn `train_test_split`; bit-identity with the
  authors' private split cannot be verified because no split files are published):
  train = first 30,816 of shuffled 2013 pool; shifted years: first 7,702 of shuffled year pool -> val/test 3,851/3,851;
  same year (2013): tail of the 2013 shuffle -> val 5,857 / test 5,858 (official behaviour; not 3,851).
* Instances: official `_make_test_examples` algorithm and RNG order, n_cands in {4, 8}; correct answer is index 0 in the official
  code, **shuffled with a fixed seed in our frozen files** (label stored).
* Name: "official-style set-to-set matching", never "official 4-choice FITB".
* Deviation (intentional): early stopping uses the SAME-YEAR validation set, never the shifted-year val set (official code uses the
  test year's val set; that is an oracle). Shifted val sets are used only for oracle diagnostics.
* Role: sanity link to the benchmark, shortcut diagnostics, secondary reliability analysis. Not the primary evidence.

### Task B — category-aligned item FITB (primary task)
* Pool: official filter + exact item-set dedupe (133 dropped; earliest `ymd, set_id` kept) across all years.
* Instance: one per eval outfit. Mask 1 random item (positive). Query = remaining items (>=3). 4 candidates = positive + 3 negatives.
* Negatives: same `category_id1` as positive; drawn slot-weighted (same marginal distribution as the positive) ONLY from outfits of
  the SAME eval set (same year, same role); not from the same user; not any item of the query outfit; 3 distinct item_ids; never
  chosen by a model score. Candidate position shuffled (fixed seed); label stored; instance_id = hash.
* Why negatives come from the same eval set: a negative drawn from train outfits would be a "seen" item (memorisation shortcut).
* Fixed + versioned: `manifests/S1_*_taskB_*.jsonl.gz`, sha256 listed. Training instances are generated online with the same rule
  from train outfits (seeded); validation/test are the frozen files.
* Not claimed: shortcut-free. Measured instead (Section 2.4). Negatives can be genuinely compatible (false negatives) -> ceiling < 100%.

## 2. Splits, leakage control, seeds

### 2.1 Project split S1 (Task B; used for the main train year 2013 and the replication train year 2014)
| role | rule | size (2013-train) |
|---|---|---|
| train | 2013 pool, users NOT in held-out set H, shuffle(seed), first 30,816 | 30,816 |
| val_id | next 3,851 of the same shuffle (model selection, calibration, thresholds ONLY here) | 3,851 |
| test_id | next 3,851 (only 3,172 left) | 3,172 |
| test_ud | outfits of held-out users H (md5(user) mod 10 == 0), never in train, random 3,851 | 3,851 |
| test_2014 / 2015 / 2017 | random 7,702 of that year's pool | 7,702 each |
Replication manifest (train 2014; test 2015/2016/2017) is frozen now, evaluated ONLY if Gate 1 passes.

### 2.2 Leakage rules (all asserted in `21_audit_and_baselines.py`)
1. Outfit-level: no set_id in two roles. 2. Exact item-set duplicates train vs any eval set = 0 (S1 by dedupe; S0 audited: 0.00-0.03%).
3. test_ud users disjoint from train users (assert). 4. Masked item never in query; negatives never in query outfit; same cat1.
5. No test-year statistic used as a training feature; feature standardisation fit on train items only. 6. Near-duplicate (cosine) audit
across train/eval is done once features exist (Gate 0 item). 7. Per-instance flags stored: user_seen, pos_seen, neg_seen.

### 2.3 Seeds
PROJECT_SEED=20261012, OFFICIAL_SEED=0; every sub-seed = md5 of (purpose, version, ids); rebuild reproduces identical file hashes (verified).
Model seeds: 0-4 (5 independent trainings; single-model results reported per seed; ensemble = all 5).

### 2.4 Measured shortcut baselines (metadata only; user-cluster 95% CI) — see `results/exp1_manifest_report.txt`
Task B (chance 25.0): cat1-only = 25.0 (by construction), cat2-prior 24.3-25.6, train-popularity 24.2-25.1, **price-near-context 27.4-30.1** (above chance).
Task A (n_cands=4, chance 25): category-complement 60-66, size+category 70-76 (n_cands=8: 52-59 vs chance 12.5). Official-style task is shortcut-dominated.

## 3. Features to extract
Item = one VGG16 fc6 vector (4096-d, fp16). Rule: union of item_ids of every outfit in every frozen set (S1 2013, S1 2014, S0 official).
Required: **391,422 items** (S1-2013 206,366; S0 207,294; S1-2014 246,555; overlaps merged) = 3.21 GB fp16 / 6.41 GB fp32.
Alternative `--all-items`: all 2,335,598 catalog items = 19.1 GB fp16. (Download volume is the same 89.5 GB either way: every tar holds a
hash-partitioned ~0.49% of every set.)
Metadata per item (no extraction needed): cat1, cat2, price, first-seen year.

## 4. Preflight checklist (status)
1 item-set rule + count frozen: DONE (391,422). 2 splits/candidates frozen before download: DONE, hashes recorded, rebuild reproducible.
3 no re-download: required list is a superset check target; `--all-items` available as insurance. 4 resumable + completed-tar log: DONE
(`done.jsonl`, curl -C -). 5 integrity: no official checksum exists (checksum URLs return a catch-all HTML 200); instead S3 multipart ETag
(16 MiB parts) is recomputed and compared -> verified on the sample tar (68d6ea07...-27 identical) + full gzip/CRC read + size check.
6 delete tar only after shard written + reloaded + counts verified: DONE. 7 post-extract coverage check (`--finalize`): required items missing
rate must be 0 or reported. 8 per-tar stats (dim, NaN/Inf, fp16 overflow, directory name): DONE; dry run on the local sample tar:
1,911 required items found (0.49% = 1/200 as expected), 0 bad. 9 license kept; features/manifests not committed, not published.
10 no image reconstruction, no identity inference. Dry run used no new download. `--run` refuses without `--authorized`.

## 5. Baseline models and evaluation flow
Features: per-dim standardisation on train items; item encoder Linear(4096,256)-GELU (shared); set = mean pooling.
* Task B: score(candidate | query) = MLP([c, e, c*e]) with c = mean-pooled query embedding. M2 visual-only; M3 = M2 + cat1 histogram of query + set size.
* Task A: x_bar, y_bar mean pools; MLP([x, y, x*y, |x-y|]); M3 adds category histograms + sizes.
* Loss: softmax cross-entropy over the candidates (4); AdamW lr 1e-3, wd 1e-2, batch 256, early stop on Val-ID NLL (patience 3), max 30 epochs.
* Reference baselines: Random; cat1/cat2/price/popularity (Section 2.4); non-learned cosine(mean query, candidate) on raw features.
Flow: train 5 seeds -> predict all frozen sets -> accuracy / NLL / Brier / ECE (equal-mass bins) / reliability diagrams -> reliability scores -> subgroup analyses.

### Reliability scores (each defined at ONE unit; never mixed)
* MSP: max softmax over the candidates of one model.
* kNN (primary) — unit = **query-candidate pair**: embedding = model penultimate vector of [query, top-1 predicted candidate]; bank = penultimate vectors of
  (train query, TRUE candidate) pairs; confidence = - mean L2 distance to k nearest bank vectors (k in {10, 50, 200} chosen on Val-ID only by error-detection AUROC).
  Secondary diagnostics, reported separately and labelled: item-level (top-1 candidate embedding vs same-cat1 train items), query-level (mean-pooled query vs train queries).
* Deep ensemble (5 members, seeds 0-4, same data): confidence variants (i) max of mean softmax, (ii) -mutual information (H(mean p) - mean H(p_m)),
  (iii) vote disagreement. Compared with single-model MSP and with the ensemble's own max probability.
* Metadata detector (control): logistic regression on information known at inference (query size, target cat1, price spread of query/candidates, candidate cat2 prior).
Error = top-1 != label. Metrics: error-detection AUROC; risk at fixed coverage (80%); risk at a FIXED threshold set on Val-ID (coverage and risk drift);
high-confidence-error rate (error rate among instances above the Val-ID 80%-coverage threshold).

### Statistics
User-cluster bootstrap (B=2000, resample users), paired across methods (same resamples). Always report n_users and effective users (Kish).
Between two disjoint sets use independent cluster bootstraps for the delta. Instance-level CIs are not reported: design effect is large
(e.g. test_ud: 3,851 instances, 593 users, effective users 53).

### Stratified shift analysis (pre-declared)
A user: seen/unseen in train (shifted years: 5-40% seen), UD vs random ID; user-capped (<=5 instances per user, fixed seed) and equal-user-weight and "drop top-10 users".
B item: positive seen/unseen (ID 71% seen; 2014 13%; 2015/2017 ~1% -> only matched through the unseen stratum: compare ID-unseen vs shifted).
C category of masked item (7 groups) and query size strata (3, 4, 5, 6+). D category-mix reweighting of ID to the shifted-year mix (optional). E visual feature shift
(after features): within-category domain classifier AUROC train-year vs test-year items (user-grouped CV) + mean kNN distance to train items.
Groups with < 50 users or effective users < 30: interval only, no interpretation.

## 6. Gates
Gate 0 (task validity, blocks everything else, NOT a hypothesis test): M2/M3 test_id accuracy CI lower bound > upper CI of best metadata baseline
(price: ~31.7%) and > cosine baseline; near-duplicate audit passes; positive-position histogram uniform (done: 24-26%).
Gate 1 (phenomenon; reference = test_ud for the same user-unseen status, test_id secondary; need >=2 of the 3 shifted years): at least one of
ECE +0.03, error-detection AUROC(MSP) -0.03, high-confidence-error rate +5 pp (absolute), with a 95% cluster-bootstrap CI excluding 0;
same direction in all 5 seeds and >= threshold in >= 3; >= 50% of the magnitude retained under user cap; placebo (two random halves of test_2014) CI covers 0.
Fail -> write a 3-day negative-result note and stop the research track.
Gate 2 (method value; >= 2 shifted years): best of {kNN-pair, ensemble-MI} beats MSP by >= 0.02 AUROC (paired CI > 0) AND beats metadata detector by >= 0.01
AND risk@80% improves >= 1.5 pp; same sign in seen-user and unseen-user strata and when any single category is left out; cost report (5x train/inference
for the ensemble, bank memory/latency for kNN). If not met: "MSP is sufficient / signals do not add" is the result.
(All thresholds are my proposals; change them BEFORE seeing results.)

## 7. Effort and time box
Extraction + finalize (1-2 h machine, ~1 h attention); features sanity + near-dup + visual shift 1 d; harness + baselines + bootstrap 4-5 d; reliability scores 3-4 d;
strata/sensitivity/figures/write-up 4-5 d = 13-16 working days (~50-65 h), 3-4 calendar weeks part-time. Gate-1 decision by day 10; hard stop at week 4.

## 8. What flows into the OutfitMatch DBMS product
Product-ready: manifest/versioning + audit tooling; scoring API contract `score(query_items, candidates) -> probs + confidence + evidence`; kNN neighbours as
evidence (similar training outfits -> Neo4j paths / Postgres rows); ensemble confidence as an API field; evaluation harness reused for Neo4j vs Postgres candidate generation.
Research-only: SHIFT15M-specific features (no images, anonymous users), Task A, shift stratification, gates, bootstrap. SHIFT15M weights are not product weights.

## 9. Amendments after the pre-training checks (written BEFORE any model is trained; manifests unchanged)
Evidence: `step0/results/exp1_check{1,2,3,4}_*.txt`.
A1. Visual familiarity is not item_id familiarity: 9.4% of train items have an exact visual twin (cosine >= 0.999, other item_id; 94% same category_id2, 90% same price).
    Item-stratified analyses use `visually_seen = id seen OR twin >= 0.999` and report NN-similarity strata, not item_id alone.
A2. Train co-occurrence route: in ID sets the positive co-occurs with a query item in a train outfit for 6-7% of instances (negatives 0.8%); an id-level
    co-occurrence-count baseline reaches 29% on test_id and 25.0-25.1% on 2014/2015/2017. This channel vanishes under shift. Report every ID result overall AND split by
    `decidable` (co-occurrence signal exists) and by the 1-2% partial-overlap (>=50% shared items) outfits (flags in results/exp1_neardup_flags_taskB.json).
A3. Add the co-occurrence-count baseline to the metadata baselines of Gate 0.
A4. Gate 0 neural criterion (replaces 'CI lower bound above the baselines' CI upper): neural test_id accuracy must exceed the cosine baseline (centred, 34.3%) with a PAIRED
    user-cluster bootstrap CI of the difference > 0, and its own CI lower bound must exceed the best metadata baseline upper CI (31.7%).
A5. Gate 1 must separate three components of any ID -> shifted-year drop: (i) training-free task drift (cosine drops ~4-5 pp even with year-matched centring),
    (ii) loss of the co-occurrence/memorisation channel, (iii) the model's own reliability change. Cosine and co-occurrence baselines are evaluated on every shifted set as reference.
A6. Task A is reference-only: highest-cosine-wins is BELOW chance (16-19%, n_cands=4) because category structure dominates; shortcut baselines reach 60-76%.
    Do not train Task A before Task B results exist; if trained, report it as a shortcut demonstration.
A7. Visual shift is real and category-controlled (domain AUROC 0.53 control -> 0.69/0.77/0.84 for 2014/2015/2017); reliability signals based on feature distance will partly be year detectors.
    Within-user AUROC (2014: 0.60, 237 users; 2015: 0.67, 59 users) shows it is not only user composition; 2017 cannot be assessed within users (too few overlapping users).
