# Experiment 2 — analysis plan (written BEFORE any reliability score was computed; 2026-10-12)

Question: with the existing frozen Task B MLPs (visual-only, 5 seeds, runs/phase2/visual_seed0-4) and frozen manifests, which cheap
reliability score best ranks the instances the model gets wrong? No new compatibility model, no new loss, no Task A, no manifest change.
Shifted-year labels are never used to choose k, thresholds, normalisation or any setting.

## Provenance of the design (honest disclosure)
* Phase 2 accuracy/ECE/confidence preview on test_id/ud/2014/2015/2017 had been seen before this plan. The choice of 2014+2015 as the MAIN years and 2017 as AUXILIARY
  was made after seeing those numbers (because of 2017's user concentration, ~53 effective users). All years are reported.
* Gate 1 and Gate 2 thresholds are those of EXPERIMENT1_SPEC.md section 6, unchanged. 2014 ECE vs test_id was already known to be +0.026 (< +0.03).

## Definitions
* Error = top-1 prediction != label (4-way). POSITIVE CLASS FOR DETECTION = ERROR. Every score is an UNCERTAINTY (higher = more likely error).
  AUROC = P(u_error > u_correct) + 0.5 P(tie). Orientation is checked on val_id (all AUROCs must be reported as-is; a value < 0.5 is a finding, not flipped).
* Predictors: single = seed 0 model (pre-declared; seeds 1-4 reported as robustness). Ensemble = mean softmax of seeds 0-4; its prediction = argmax of the mean.
* Two comparison families (the error labels differ between a single model and an ensemble, so they are never mixed in one AUROC table):
  S-family (error of seed-0 model): msp_single, knn_pair, ens_msp, ens_entropy, ens_mi, ens_vote_peers (disagreement of seeds 1-4 with seed-0's prediction), meta.
  E-family (error of the ensemble): ens_msp, ens_entropy, ens_mi, ens_vote, msp_single, knn_pair, meta.
  Paired reference: S-family vs msp_single; E-family vs ens_msp (so ensemble gains from better probabilities are not credited to disagreement).
* Scores
  msp_single = -max softmax (seed 0). ens_msp = -max of mean softmax. ens_entropy = H(mean p). ens_mi = H(mean p) - mean_m H(p_m). ens_vote = 1 - share of members whose argmax equals the ensemble prediction.
  knn_pair (unit = QUERY-PREDICTED-CANDIDATE PAIR): embedding = 256-d hidden activation of the scorer MLP for the pair (query, top-1 predicted candidate), L2-normalised;
  bank = same embedding for (train query, TRUE candidate) pairs, one fixed draw per train outfit (rng 4242), 30,816 vectors; score = mean Euclidean distance to the k nearest bank vectors;
  k in {10, 50, 200} chosen by val_id error-detection AUROC only. No per-year normalisation of any kind.
  meta (control) = L2 logistic regression on inference-time metadata only (query size, target cat1, query/candidate price statistics, predicted candidate's cat2 prior), fit on val_id
  (cross-fitted for val_id itself), then applied unchanged.
* Metrics: AUROC; risk@80% coverage with each set's OWN ranking (ranking ability); fixed-threshold deployment: tau = the uncertainty value with 80% coverage on val_id, applied unchanged to every
  set -> coverage and risk (= high-confidence-error rate among accepted) and accepted-and-wrong share of all instances. Own-ranking and fixed-threshold are reported in separate tables.
  Calibration (NLL, Brier, ECE with 15 equal-mass bins), confidence distribution, runtime, memory.
* Uncertainty: user-cluster bootstrap B=2000 (users resampled with all their instances), the SAME resamples for all methods of a set (paired). Between two disjoint sets independent resamples.
  Effective users (Kish) reported. User-capped sensitivity: <=5 instances per user per set (fixed draw seed 0 for the interval; 20 draws for stability).

## CONFIRMATORY (pre-declared in the spec; exactly these)
C1 Gate 1 (spec section 6), for each shifted year 2014/2015/2017 against test_ud (primary reference) and test_id (secondary), single-model MSP:
   dECE >= +0.03 OR dAUROC(MSP) <= -0.03 OR dHCE >= +5 pp (HCE = risk among accepted at the val_id 80%-coverage threshold), 95% user-cluster CI excluding 0 in the right direction;
   same direction in all 5 seeds and >= threshold in >= 3 seeds; >= 50% of the magnitude retained under the user cap; placebo (random user-halves of test_2014) CI covers 0.
   Gate 1 passes only if >= 2 of the 3 shifted years satisfy it.
C2 Gate 2 (spec section 6), >= 2 shifted years: knn_pair vs msp_single (S-family) and ens_mi vs ens_msp (E-family) each: paired dAUROC >= +0.02 (CI > 0) AND beats meta by >= 0.01 (CI > 0)
   AND risk@80% (own ranking) improves >= 1.5 pp vs the family MSP AND same sign in seen-user and unseen-user strata and in all leave-one-category-out analyses; cost reported.
   (Two candidate methods are tested; I report both and do not count a pass for one as a pass for the family.)
If C1 fails the strict conditions, everything below is labelled EXPLORATORY and the report says so.

## EXPLORATORY (everything else)
entropy/vote scores; seeds 1-4 consistency; novel-and-clean subgroup; strata by seen/twin/novel positives; kNN-vs-year diagnostics (year-detection AUROC, correlation with MSP);
candidate-only/query-only kNN variants (not run unless needed); fixed-threshold results; 2014+2015 emphasis; any model-vs-model cost judgement.
Linear baseline supplement is done AFTER this experiment, train/val_id only.

## Wording constraints
"descriptive baseline-relative degradation" (not causal decomposition). No claim that temporal shift CAUSES calibration degradation; no claim the model "knows its errors" unless C2 is met.
