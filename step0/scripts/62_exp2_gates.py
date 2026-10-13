"""Experiment 2: formal Gate 1 and Gate 2 exactly as pre-declared (EXPERIMENT1_SPEC.md section 6, EXPERIMENT2_PREREG.md). Thresholds unchanged.
PYTHONPATH=./pylibs:./scripts python3 -u scripts/62_exp2_gates.py 2>&1 | tee results/exp2_gates_report.txt
"""
import pickle
import numpy as np
from exp2_data import *

B = 2000
THR = {"ECE": 0.03, "AUROC": -0.03, "HCE": 0.05}       # ECE up, AUROC down, high-confidence-error rate up
YEARS = ["test_2014", "test_2015", "test_2017"]
Ps = {s: {r: P[s][r] for r in ROLES} for s in range(5)}


def tau_seed(s):
    u = -Ps[s]["val_id"].max(1); return np.sort(u)[int(np.ceil(0.8 * len(u))) - 1]


TAUS = {s: tau_seed(s) for s in range(5)}


class GSet:
    """per-seed single-model MSP metrics (ECE, AUROC, HCE) for one set (optionally a subset)."""
    def __init__(self, role, mask=None):
        m = np.ones(len(label[role]), bool) if mask is None else mask
        self.users = users[role][m]; self.n = int(m.sum()); self.objs = []
        for s in range(5):
            p = Ps[s][role][m]; y = label[role][m]; err = (p.argmax(1) != y).astype(float); u = -p.max(1)
            self.objs.append((Scorer(u, err), Ece(p.max(1), 1 - err), (u <= TAUS[s]).astype(float), err))

    def metrics(self, w):
        out = np.empty((5, 3))
        for s, (S, E, acc, err) in enumerate(self.objs):
            wa = w * acc
            out[s] = (E(w), S.auc(w), (wa * err).sum() / wa.sum() if wa.sum() > 0 else np.nan)
        return out

    def boot(self, seed):
        W = boot_weights(self.users, B, seed)
        return np.stack([self.metrics(W[b].astype(float)) for b in range(B)])        # [B,5,3]


def point(g):
    return g.metrics(np.ones(g.n))


H("GATE 1 (confirmatory, thresholds unchanged): single-model MSP; metrics averaged over the 5 seed models")
print("metrics per seed: ECE (15 equal-mass bins), error-detection AUROC of MSP, HCE = error rate among accepted at the val_id 80%-coverage threshold (per seed).")
print("delta = shifted - reference. Thresholds: dECE >= +0.03 | dAUROC <= -0.03 | dHCE >= +0.05. CIs: independent user-cluster bootstraps of the two sets (B=2000).")
full = {r: GSet(r) for r in EVAL}
Mb = {r: full[r].boot(seed_of("gate1", r)) for r in EVAL}
Mp = {r: point(full[r]) for r in EVAL}
print("\nset-level metrics (seed-mean point; per-seed range):")
print(f"{'set':<10}{'ECE':>22}{'AUROC(MSP)':>24}{'HCE %':>22}{'coverage at tau %':>20}")
for r in EVAL:
    cov = np.mean([(-Ps[s][r].max(1) <= TAUS[s]).mean() for s in range(5)])
    print(f"{r:<10}" + "".join(f"{Mp[r][:, j].mean()*(100 if j==2 else 1):>12.3f} [{Mp[r][:, j].min()*(100 if j==2 else 1):.3f},{Mp[r][:, j].max()*(100 if j==2 else 1):.3f}]" for j in range(3)) + f"{cov*100:>20.1f}")
# user cap: 20 draws point, draw 0 bootstrap
caps = {}
for r in EVAL:
    pts = []
    for d_ in range(20):
        pts.append(point(GSet(r, cap_mask(users[r], 5, 100 + d_))).mean(0))
    g0 = GSet(r, cap_mask(users[r], 5, 100)); caps[r] = (np.mean(pts, 0), g0.boot(seed_of("gate1cap", r)))
# placebo: random halves of test_2014 by USER
u14 = np.unique(users["test_2014"]); rs = np.random.RandomState(0); perm = rs.permutation(u14)
A_u = set(perm[:len(perm) // 2]); mA = np.array([u in A_u for u in users["test_2014"]])
gA, gB = GSet("test_2014", mA), GSet("test_2014", ~mA)
Pd = (point(gA) - point(gB)).mean(0); Pb = gA.boot(seed_of("plA")).mean(1) - gB.boot(seed_of("plB")).mean(1)
dist = []
for k in range(200):
    pm = np.random.RandomState(1000 + k).permutation(u14); Au = set(pm[:len(pm) // 2]); m_ = np.array([u in Au for u in users["test_2014"]])
    dist.append((point(GSet("test_2014", m_)) - point(GSet("test_2014", ~m_))).mean(0))
dist = np.array(dist)
print(f"\nPLACEBO (random user-halves of test_2014; halves of {gA.n}/{gB.n} instances): split-0 delta ECE {Pd[0]:+.3f} [{ci(Pb[:,0])[0]:+.3f},{ci(Pb[:,0])[1]:+.3f}], "
      f"AUROC {Pd[1]:+.3f} [{ci(Pb[:,1])[0]:+.3f},{ci(Pb[:,1])[1]:+.3f}], HCE {Pd[2]*100:+.1f}pp [{ci(Pb[:,2])[0]*100:+.1f},{ci(Pb[:,2])[1]*100:+.1f}]")
print(f"   200 random splits, 95% range of delta: ECE [{np.percentile(dist[:,0],2.5):+.3f},{np.percentile(dist[:,0],97.5):+.3f}], AUROC [{np.percentile(dist[:,1],2.5):+.3f},{np.percentile(dist[:,1],97.5):+.3f}], HCE pp [{np.percentile(dist[:,2],2.5)*100:+.1f},{np.percentile(dist[:,2],97.5)*100:+.1f}]")
pl_cover = [ci(Pb[:, j])[0] <= 0 <= ci(Pb[:, j])[1] for j in range(3)]

names_m = ["ECE", "AUROC", "HCE"]
verdict = {}
for ref in ("test_ud", "test_id"):
    print("\n" + "=" * 8, f"reference = {ref} ({'PRIMARY' if ref=='test_ud' else 'secondary'})", "=" * 8)
    for y in YEARS:
        Db = Mb[y] - Mb[ref]; Dp = Mp[y] - Mp[ref]
        capd = caps[y][0] - caps[ref][0]; capb = caps[y][1] - caps[ref][1]
        for j, nm in enumerate(names_m):
            thr = THR[nm]; sgn = 1 if thr > 0 else -1
            d_mean = Dp[:, j].mean(); d_bs = Db[:, j, :].mean(1) if False else Db[:, :, j].mean(1)
            lo, hi = ci(d_bs)
            c_point = sgn * d_mean >= abs(thr)
            c_ci = (lo > 0) if sgn > 0 else (hi < 0)
            c_dir = bool(np.all(sgn * Dp[:, j] > 0)); c_n = int((sgn * Dp[:, j] >= abs(thr)).sum()) >= 3
            ret = (capd[j] / d_mean) if d_mean != 0 else np.nan
            c_cap = bool(ret >= 0.5)
            c_pl = bool(pl_cover[j])
            ok = all([c_point, c_ci, c_dir, c_n, c_cap, c_pl]); verdict[(ref, y, nm)] = ok
            sc_ = 100 if nm == "HCE" else 1
            print(f"{y:<10}{nm:<6} delta={d_mean*sc_:+.3f}{'pp' if nm=='HCE' else '  '} [{lo*sc_:+.3f},{hi*sc_:+.3f}] | thr {thr*sc_:+.2f} | per-seed " +
                  " ".join(f"{v*sc_:+.3f}" for v in Dp[:, j]) + f" | point>=thr:{c_point} CI:{c_ci} dir5/5:{c_dir} >=3seeds:{c_n} cap-retention {ret*100:.0f}%:{c_cap} placebo-covers-0:{c_pl} => {'PASS' if ok else 'fail'}")
    for_year = {y: any(verdict[(ref, y, nm)] for nm in names_m) for y in YEARS}
    print(f"  years passing at least one criterion with ALL conditions: {[y for y,v in for_year.items() if v]}  -> Gate 1 ({ref}) {'PASSES' if sum(for_year.values()) >= 2 else 'DOES NOT PASS'} (needs >=2 of 3 years)")

# =============================================================================================== GATE 2
H("GATE 2 (confirmatory, thresholds unchanged): needs >= 2 of 3 shifted years")
RES_ = pickle.load(open(os.path.join(EXP2, "main_bootstrap.pkl"), "rb"))
cands = [("knn_pair", "S", "msp_single"), ("ens_mi", "E", "ens_msp")]
cat_of = {r: T(r, "cat1") for r in EVAL}
g2 = {}
for (name, fam, ref) in cands:
    print(f"\n--- candidate {name} ({fam}-family, reference {ref}) ---")
    for y in YEARS:
        res = RES_[(fam, y)][0]
        d_ref = res[name][1]["auc"] - res[ref][1]["auc"]; p_ref = res[name][0]["auc"] - res[ref][0]["auc"]
        d_meta = res[name][1]["auc"] - res["meta"][1]["auc"]; p_meta = res[name][0]["auc"] - res["meta"][0]["auc"]
        rk = res[ref][1]["risk80"] - res[name][1]["risk80"]; p_rk = res[ref][0]["risk80"] - res[name][0]["risk80"]
        a = (p_ref >= 0.02) and ci(d_ref)[0] > 0
        b = (p_meta >= 0.01) and ci(d_meta)[0] > 0
        c = (p_rk >= 0.015)
        # strata: seen/unseen user, leave-one-category-out (point sign of dAUROC vs reference)
        f_ = FAM[fam]; U = SC[y][f_["idx"]]; err = f_["err"][y]
        def dpt(m):
            return auc_pt(U[name][m], err[m]) - auc_pt(U[ref][m], err[m])
        signs, notes = [], []
        for lab, m in (("user-seen", T(y, "user_seen").astype(bool)), ("user-unseen", ~T(y, "user_seen").astype(bool))):
            if m.sum() >= 50 and len(set(users[y][m])) >= 20:
                signs.append(dpt(m) > 0); notes.append(f"{lab} {dpt(m):+.3f}")
            else:
                notes.append(f"{lab} n/a")
        for ctg in range(10, 17):
            m = cat_of[y] != ctg; signs.append(dpt(m) > 0)
        d_ = bool(all(signs))
        ok = a and b and c and d_; g2[(name, y)] = ok
        print(f"{y:<10} dAUROC vs {ref}: {p_ref:+.3f} [{ci(d_ref)[0]:+.3f},{ci(d_ref)[1]:+.3f}] (need >=+0.02, CI>0: {a}) | vs meta: {p_meta:+.3f} [{ci(d_meta)[0]:+.3f},{ci(d_meta)[1]:+.3f}] (need >=+0.01, CI>0: {b}) | "
              f"risk@80 improvement {p_rk*100:+.2f}pp [{ci(rk)[0]*100:+.2f},{ci(rk)[1]*100:+.2f}] (need >=+1.5: {c}) | same sign in strata/LOCO: {d_} ({'; '.join(notes)}; LOCO signs positive {sum(signs[len(signs)-7:])}/7) => {'PASS' if ok else 'fail'}")
    n_ok = sum(g2[(name, y)] for y in YEARS)
    print(f"  => {name}: passes in {n_ok}/3 years -> Gate 2 {'PASSES' if n_ok >= 2 else 'DOES NOT PASS'} for this candidate")
print("\nCost (for the record): single 1x; ensemble 5.1x forward, 5x parameters and training; kNN 1.4x forward + 31.6 MB bank (see main report TABLE 10).")
H("SUMMARY")
g1_ud = [y for y in YEARS if any(verdict[("test_ud", y, nm)] for nm in names_m)]; g1_id = [y for y in YEARS if any(verdict[("test_id", y, nm)] for nm in names_m)]
print(f"Gate 1 (ref test_ud, primary): years passing = {g1_ud} -> {'PASS' if len(g1_ud)>=2 else 'NOT PASSED'}; (ref test_id, secondary): years passing = {g1_id} -> {'PASS' if len(g1_id)>=2 else 'NOT PASSED'}")
print("Gate 2: " + "; ".join(f"{n}: {sum(g2[(n, y)] for y in YEARS)}/3 years" for n, _, _ in cands) + " -> " + ("PASS" if any(sum(g2[(n, y)] for y in YEARS) >= 2 for n, _, _ in cands) else "NOT PASSED"))
