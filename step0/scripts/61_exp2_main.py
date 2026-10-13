"""Experiment 2 main comparison (frozen models, frozen manifests). Run:
PYTHONPATH=./pylibs:./scripts python3 -u scripts/61_exp2_main.py 2>&1 | tee results/exp2_main_report.txt
"""
import pickle, time, json
import numpy as np
from scipy.stats import spearmanr
from exp2_data import *

B = 2000
f3 = lambda x: f"{x:6.3f}"
def cis(pt, arr, d=3, sc_=1.0):
    lo, hi = ci(arr); return f"{pt*sc_:.{d}f} [{lo*sc_:.{d}f},{hi*sc_:.{d}f}]"
def dci(a, b, d=3, sc_=1.0):          # paired difference of bootstrap arrays a-b with point from arrays' point values
    return a, b


# ====================================================================================== setup
H("0. SETUP / SANITY")
print("positive class = ERROR (top-1 != label); every score is an UNCERTAINTY (higher = more likely error). AUROC>0.5 means the score ranks errors above correct predictions.")
print(f"k selection for knn_pair on val_id ONLY (error-detection AUROC of each seed's own errors): " + "; ".join(f"seed{s}: k={K_SEL[s][0]} ({', '.join(f'{k}:{v:.3f}' for k, v in K_SEL[s][1].items())})" for s in range(5)))
print("val_id thresholds (tau, coverage 80%):", {f"{f}/{n}": round(float(t), 3) for (f, n), t in TAU.items() if f == "S"})
print(f"\n{'set':<10}{'n':>6}{'users':>7}{'eff.users':>10}{'acc seed0':>11}{'acc ens':>9}{'acc mean5':>10}{'err seed0':>10}{'err ens':>8}")
for r in ROLES:
    a5 = np.mean([(P[s][r].argmax(1) == label[r]).mean() for s in range(5)])
    print(f"{r:<10}{len(label[r]):>6}{len(set(users[r])):>7}{eff_users(users[r])[0]:>10.0f}{(1-err_s[r].mean())*100:>11.1f}{(1-err_e[r].mean())*100:>9.1f}{a5*100:>10.1f}{err_s[r].mean()*100:>10.1f}{err_e[r].mean()*100:>8.1f}")
print("\nval_id AUROC (orientation/sanity): S-family " + ", ".join(f"{n}={auc_pt(u, err_s['val_id']):.3f}" for n, u in SC['val_id'][0].items()))
print("                                   E-family " + ", ".join(f"{n}={auc_pt(u, err_e['val_id']):.3f}" for n, u in SC['val_id'][1].items()))

# ====================================================================================== main bootstrap
t0 = time.time(); RES_ = {}
for fam in FAM:
    for r in EVAL:
        RES_[(fam, r)] = analyze(fam, r, B=B)
    print(f"[bootstrap family {fam} done {time.time()-t0:.0f}s]", flush=True)
pickle.dump(RES_, open(os.path.join(EXP2, "main_bootstrap.pkl"), "wb"))
names = {"S": list(SC["val_id"][0]), "E": list(SC["val_id"][1])}

for fam, title in (("S", "S-FAMILY: errors of the SEED-0 single model (reference = msp_single)"), ("E", "E-FAMILY: errors of the 5-SEED ENSEMBLE (reference = ens_msp)")):
    ref = FAM[fam]["ref"]
    H(f"TABLE 1 [{title}] error-detection AUROC [user-cluster 95% CI]; chance 0.5")
    print(f"{'method':<16}" + "".join(f"{r:>24}" for r in EVAL))
    for n in names[fam]:
        print(f"{n:<16}" + "".join(f"{cis(RES_[(fam, r)][0][n][0]['auc'], RES_[(fam, r)][0][n][1]['auc']):>24}" for r in EVAL))
    H(f"TABLE 2 [{fam}] PAIRED difference in AUROC vs {ref} [paired user-cluster 95% CI]")
    print(f"{'method':<16}" + "".join(f"{r:>24}" for r in EVAL))
    for n in names[fam]:
        if n == ref:
            continue
        row = f"{n:<16}"
        for r in EVAL:
            res = RES_[(fam, r)][0]; d = res[n][1]["auc"] - res[ref][1]["auc"]; p = res[n][0]["auc"] - res[ref][0]["auc"]
            row += f"{cis(p, d):>24}"
        print(row)
    H(f"TABLE 3 [{fam}] RISK at 80% coverage, each set's OWN ranking (error rate among the 80% most confident) [CI]; base = error rate at 100% coverage")
    print(f"{'method':<16}" + "".join(f"{r:>24}" for r in EVAL))
    print(f"{'(base error)':<16}" + "".join(f"{cis(RES_[(fam, r)][1]['err'], RES_[(fam, r)][2], 1, 100):>24}" for r in EVAL))
    for n in names[fam]:
        print(f"{n:<16}" + "".join(f"{cis(RES_[(fam, r)][0][n][0]['risk80'], RES_[(fam, r)][0][n][1]['risk80'], 1, 100):>24}" for r in EVAL))
    print(f"\n  improvement vs {ref} (pp lower risk = positive) [paired CI]")
    for n in names[fam]:
        if n == ref:
            continue
        row = f"{n:<16}"
        for r in EVAL:
            res = RES_[(fam, r)][0]; d = res[ref][1]["risk80"] - res[n][1]["risk80"]; p = res[ref][0]["risk80"] - res[n][0]["risk80"]
            row += f"{cis(p, d, 2, 100):>24}"
        print(row)
    H(f"TABLE 4 [{fam}] FIXED val_id THRESHOLD (tau = uncertainty at 80% coverage on val_id, applied unchanged). coverage % / risk among accepted % (= high-confidence-error rate) [CI]")
    print(f"{'method':<16}{'val cov%':>9}" + "".join(f"{r:>34}" for r in EVAL))
    for n in names[fam]:
        u = SC["val_id"][FAM[fam]["idx"]][n]; vc = (u <= TAU[(fam, n)]).mean() * 100
        row = f"{n:<16}{vc:>9.1f}"
        for r in EVAL:
            pt, bs = RES_[(fam, r)][0][n]
            row += f"{cis(pt['cov'], bs['cov'], 1, 100):>17} {cis(pt['risk_fix'], bs['risk_fix'], 1, 100):>16}"
        print(row)
    print(f"\n  accepted-AND-wrong as % of ALL instances (fixed threshold):")
    for n in names[fam]:
        print(f"{n:<16}{'':>9}" + "".join(f"{cis(RES_[(fam, r)][0][n][0]['hce_all'], RES_[(fam, r)][0][n][1]['hce_all'], 1, 100):>34}" for r in EVAL))

# ====================================================================================== deployment view
H("TABLE 5. DEPLOYMENT VIEW across predictors (each row = predictor + its score): accuracy, risk@80% own ranking, fixed val_id threshold -> coverage / risk")
rows_ = [("single + MSP", "S", "msp_single"), ("single + kNN-pair", "S", "knn_pair"), ("ensemble + MSP", "E", "ens_msp"), ("ensemble + MI", "E", "ens_mi"),
         ("ensemble + entropy", "E", "ens_entropy"), ("ensemble + vote", "E", "ens_vote")]
print(f"{'system':<22}{'metric':<12}" + "".join(f"{r:>12}" for r in EVAL))
for nm, fam, n in rows_:
    for met in ("acc%", "risk80%", "cov_fix%", "risk_fix%"):
        vals = []
        for r in EVAL:
            pt = RES_[(fam, r)][0][n][0]
            vals.append({"acc%": (1 - RES_[(fam, r)][1]["err"]) * 100, "risk80%": pt["risk80"] * 100, "cov_fix%": pt["cov"] * 100, "risk_fix%": pt["risk_fix"] * 100}[met])
        print(f"{nm if met=='acc%' else '':<22}{met:<12}" + "".join(f"{v:>12.1f}" for v in vals))

# ====================================================================================== risk-coverage numbers + figure
H("TABLE 6. Risk-coverage (own ranking), risk % at coverage 50/60/70/80/90/100 (point estimates)")
covs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
print(f"{'set':<10}{'system':<22}" + "".join(f"{int(c*100):>8}" for c in covs))
for r in EVAL:
    for nm, fam, n in rows_:
        u = SC[r][FAM[fam]["idx"]][n]; cv = Scorer(u, FAM[fam]["err"][r]).curve(np.ones(len(u)), covs)
        print(f"{r:<10}{nm:<22}" + "".join(f"{v*100:>8.1f}" for v in cv))
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(os.path.join(EXP2, "figures"), exist_ok=True)
    grid = np.linspace(0.1, 1.0, 19)
    fig, ax = plt.subplots(2, 5, figsize=(20, 7), sharey="row")
    for ci_, r in enumerate(EVAL):
        for ri, fam in enumerate(("S", "E")):
            for n in names[fam]:
                u = SC[r][FAM[fam]["idx"]][n]; cv = Scorer(u, FAM[fam]["err"][r]).curve(np.ones(len(u)), grid)
                ax[ri, ci_].plot(grid * 100, cv * 100, label=n, lw=2 if n == FAM[fam]["ref"] else 1)
            ax[ri, ci_].axhline(FAM[fam]["err"][r].mean() * 100, ls=":", c="gray"); ax[ri, ci_].set_title(f"{r} [{fam}-family]", fontsize=9); ax[ri, ci_].set_xlabel("coverage %")
    ax[0, 0].set_ylabel("risk % (S: seed-0 errors)"); ax[1, 0].set_ylabel("risk % (E: ensemble errors)"); ax[0, 0].legend(fontsize=6); ax[1, 0].legend(fontsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(EXP2, "figures", "risk_coverage.png"), dpi=110)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for k, (fam, n) in enumerate([("S", "msp_single"), ("S", "knn_pair"), ("E", "ens_msp"), ("E", "ens_mi")]):
        ax[0].plot(range(5), [RES_[(fam, r)][0][n][0]["cov"] * 100 for r in EVAL], "o-", label=f"{fam}:{n}")
        ax[1].plot(range(5), [RES_[(fam, r)][0][n][0]["risk_fix"] * 100 for r in EVAL], "o-", label=f"{fam}:{n}")
    for a, t in zip(ax, ["coverage % at val_id tau (target 80)", "risk % among accepted at val_id tau"]):
        a.set_xticks(range(5)); a.set_xticklabels(EVAL, fontsize=7); a.set_title(t, fontsize=9)
    ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(os.path.join(EXP2, "figures", "fixed_threshold.png"), dpi=110)
    print("saved runs/exp2/figures/risk_coverage.png, fixed_threshold.png")
except Exception as ex:
    print("figures skipped:", ex)

# ====================================================================================== calibration / confidence distribution
H("TABLE 7. Calibration and confidence distribution (point estimates). seed0 = single model; ens = mean softmax; mean5 = average over the 5 single models")
def ece_pt(p, y):
    c = p.max(1); ok = (p.argmax(1) == y).astype(float); return Ece(c, ok)(np.ones(len(y)))
def brier(p, y):
    oh = np.eye(4)[y]; return ((p - oh) ** 2).sum(1).mean()
def nll(p, y):
    return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
print(f"{'':<10}{'model':<8}{'acc%':>6}{'NLL':>7}{'Brier':>7}{'ECE':>7}{'conf-acc pp':>12}{'maxp q10/25/50/75/90':>30}{'share maxp>=0.8':>16}")
for r in ROLES:
    for nm, p in (("seed0", P[0][r]), ("ens", PE[r])):
        y = label[r]; c = p.max(1)
        print(f"{r if nm=='seed0' else '':<10}{nm:<8}{(p.argmax(1)==y).mean()*100:>6.1f}{nll(p,y):>7.3f}{brier(p,y):>7.3f}{ece_pt(p,y):>7.3f}{(c.mean()-(p.argmax(1)==y).mean())*100:>12.1f}"
              f"{' '.join(f'{np.percentile(c,q):.2f}' for q in (10,25,50,75,90)):>30}{(c>=0.8).mean()*100:>15.1f}%")
    ms = [(nll(P[s][r], label[r]), brier(P[s][r], label[r]), ece_pt(P[s][r], label[r])) for s in range(5)]
    print(f"{'':<10}{'mean5':<8}{'':>6}{np.mean([m[0] for m in ms]):>7.3f}{np.mean([m[1] for m in ms]):>7.3f}{np.mean([m[2] for m in ms]):>7.3f}")

# ====================================================================================== kNN diagnostics
H("TABLE 8. Does kNN reflect YEAR / DENSITY rather than error? (exploratory diagnostics, point estimates)")
print("mean kNN distance (k=10 bank distance of the predicted pair) and mean MSP per set -> density shift is real:")
for r in ROLES:
    print(f"  {r:<10} mean knn dist seed0 = {SC[r][0]['knn_pair'].mean():.3f} | mean maxp = {P[0][r].max(1).mean():.3f}")
print("\nYear-detection AUROC (instance level, no training): how well the score separates test_id (0) from the shifted set (1); 0.5 = cannot tell")
print(f"{'score':<14}" + "".join(f"{r:>12}" for r in ("test_2014", "test_2015", "test_2017")))
for n in ("msp_single", "knn_pair", "ens_mi"):
    row = f"{n:<14}"
    for r in ("test_2014", "test_2015", "test_2017"):
        u0, u1 = SC["test_id"][0][n], SC[r][0][n]; u = np.r_[u0, u1]; y = np.r_[np.zeros(len(u0)), np.ones(len(u1))]
        row += f"{Scorer(u, y).auc(np.ones(len(u))):>12.3f}"
    print(row)
print("\nPooled error AUROC (test_id + shifted year pooled; seed-0 errors) vs the WITHIN-set error AUROC; pooled >> within means the score is partly a year detector")
print(f"{'score':<14}" + "".join(f"{'pooled ID+'+r[5:]:>16}{'within-'+r[5:]:>12}" for r in ("test_2014", "test_2015", "test_2017")))
for n in ("msp_single", "knn_pair", "ens_mi"):
    row = f"{n:<14}"
    for r in ("test_2014", "test_2015", "test_2017"):
        u = np.r_[SC["test_id"][0][n], SC[r][0][n]]; e = np.r_[err_s["test_id"], err_s[r]]
        row += f"{auc_pt(u, e):>16.3f}{auc_pt(SC[r][0][n], err_s[r]):>12.3f}"
    print(row)
print("\nSpearman correlation of knn_pair with msp_single WITHIN each set (1 = same ranking):")
print("  " + ", ".join(f"{r}: {spearmanr(SC[r][0]['knn_pair'], SC[r][0]['msp_single'])[0]:.3f}" for r in ROLES))
print("\nPER-SEED consistency (each seed: its own errors, its own k chosen on val_id): AUROC knn_pair - AUROC msp (point) for seeds 0..4")
print(f"{'set':<10}" + "".join(f"{'seed'+str(s):>10}" for s in range(5)) + f"{'mean':>8}   (msp AUROC mean over seeds)")
for r in EVAL:
    ds, ms_ = [], []
    for s in range(5):
        e = (P[s][r].argmax(1) != label[r]).astype(float)
        a_k = auc_pt(sc[f"s{s}/{r}/knn{K_SEL[s][0]}"], e); a_m = auc_pt(-P[s][r].max(1), e); ds.append(a_k - a_m); ms_.append(a_m)
    print(f"{r:<10}" + "".join(f"{d:>10.3f}" for d in ds) + f"{np.mean(ds):>8.3f}   ({np.mean(ms_):.3f})")

# ====================================================================================== subgroups, user cap (exploratory)
H("TABLE 9. SUBGROUPS and USER-CAP sensitivity (exploratory). AUROC [CI] and paired difference vs family reference; B=1000")
sel = {"S": ["msp_single", "knn_pair", "ens_mi"], "E": ["ens_msp", "ens_mi", "knn_pair"]}
def sub_report(label_, masks):
    print(f"\n-- {label_}")
    print(f"{'set':<10}{'n':>6}{'users':>7}{'err%':>6} | " + " | ".join(f"{('S:'+n):<26}" for n in sel['S']) + " | " + " | ".join(f"{('E:'+n):<26}" for n in sel['E']))
    for r in EVAL:
        m = masks[r]
        if m.sum() < 80 or len(set(users[r][m])) < 20:
            print(f"{r:<10}{int(m.sum()):>6}  (too few instances/users)"); continue
        out = {}
        for fam in ("S", "E"):
            res, base, bb, n_, nu = analyze(fam, r, B=1000, mask=m, methods=sel[fam])
            ref = FAM[fam]["ref"]
            for n in sel[fam]:
                pt, bs = res[n]
                if n == ref:
                    out[(fam, n)] = f"{pt['auc']:.3f} [{ci(bs['auc'])[0]:.3f},{ci(bs['auc'])[1]:.3f}]"
                else:
                    d = bs["auc"] - res[ref][1]["auc"]; p = pt["auc"] - res[ref][0]["auc"]; lo, hi = ci(d)
                    out[(fam, n)] = f"{pt['auc']:.3f} d={p:+.3f} [{lo:+.3f},{hi:+.3f}]"
        print(f"{r:<10}{int(m.sum()):>6}{len(set(users[r][m])):>7}{err_s[r][m].mean()*100:>6.1f} | " + " | ".join(f"{out[('S', n)]:<26}" for n in sel['S']) + " | " + " | ".join(f"{out[('E', n)]:<26}" for n in sel['E']))
clean = {r: (T(r, "stratum") == 2) & ~T(r, "decidable").astype(bool) & ~T(r, "id_leak50").astype(bool) & ~T(r, "nd_flagged").astype(bool) for r in EVAL}
sub_report("ALL instances (reference)", {r: np.ones(len(label[r]), bool) for r in EVAL})
sub_report("visually NOVEL positive AND clean (no co-occurrence signal / overlap / near-dup flag)", clean)
sub_report("user SEEN in train", {r: T(r, "user_seen").astype(bool) for r in EVAL})
sub_report("user UNSEEN in train", {r: ~T(r, "user_seen").astype(bool) for r in EVAL})
sub_report("positive id-SEEN (item_id in train)", {r: T(r, "stratum") == 0 for r in EVAL})
sub_report("positive visually novel (any)", {r: T(r, "stratum") == 2 for r in EVAL})
capm = {r: cap_mask(users[r], 5, 0) for r in EVAL}
sub_report("USER-CAPPED: <=5 instances per user (draw seed 0)", capm)
print("\nUser-cap stability over 20 random capped draws: point AUROC range (min-max) for S:msp_single / S:knn_pair / E:ens_msp / E:ens_mi")
for r in EVAL:
    a = {k: [] for k in [("S", "msp_single"), ("S", "knn_pair"), ("E", "ens_msp"), ("E", "ens_mi")]}
    for d_ in range(20):
        m = cap_mask(users[r], 5, 100 + d_)
        for (fam, n) in a:
            a[(fam, n)].append(auc_pt(SC[r][FAM[fam]["idx"]][n][m], FAM[fam]["err"][r][m]))
    print(f"  {r:<10}" + "  ".join(f"{fam}:{n} {min(v):.3f}-{max(v):.3f}" for (fam, n), v in a.items()) + f"   (n kept ~{int(cap_mask(users[r], 5, 100).sum())})")

# ====================================================================================== cost
H("TABLE 10. COMPUTE / MEMORY (RTX 3070 Ti, test_2015 = 7,702 instances, features already on GPU)")
c = json.load(open(os.path.join(EXP2, "cost.json"))); n_ = c["n_instances"]
print(f"single model forward           : {c['single_forward_s']*1e3:7.1f} ms  ({c['single_forward_s']/n_*1e6:.2f} us/instance)  | params {c['params_single']:,} | train {c['train_seconds_single']} s")
print(f"ensemble (5 models) forward    : {c['ensemble5_forward_s']*1e3:7.1f} ms  ({c['ensemble5_forward_s']/c['single_forward_s']:.1f}x single)           | params {c['params_ensemble']:,} (5x) | train {c['train_seconds_ensemble']:.1f} s (5x)")
print(f"kNN-pair total (forward+hidden+search over {c['bank_bytes']//(256*4):,}-vector bank): {c['knn_total_s']*1e3:7.1f} ms ({c['knn_total_s']/c['single_forward_s']:.1f}x single) | bank memory {c['bank_bytes']/1e6:.1f} MB | one-off bank build {c['bank_build_s']*1e3:.0f} ms | extra training: none")
print("(ensemble MSP/entropy/MI/vote cost the same 5x; kNN adds a stored bank of train activations and a search; no extra training for either kNN or MSP)")
