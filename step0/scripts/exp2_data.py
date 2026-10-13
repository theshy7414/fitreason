"""Experiment 2: assemble all uncertainty scores (S-family: errors of the seed-0 model; E-family: errors of the 5-seed ensemble)."""
import json, collections
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from exp1_common import *
from exp2_lib import *

EXP2 = os.path.join(ROOT, "runs", "exp2")
ROLES = ["val_id", "test_id", "test_ud", "test_2014", "test_2015", "test_2017"]
EVAL = ROLES[1:]
KS = (10, 50, 200)
sc = np.load(os.path.join(EXP2, "scores.npz"))
tab = np.load(os.path.join(RES, "exp1_instance_table_taskB.npz"))
T = lambda r, k: tab[f"{r}/{k}"]
label = {r: T(r, "label") for r in ROLES}
users = {r: T(r, "user_id") for r in ROLES}


def softmax(l):
    e = np.exp(l - l.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def entropy(p):
    return -(p * np.log(p + 1e-12)).sum(1)


P = {s: {r: softmax(sc[f"s{s}/{r}/logits"]) for r in ROLES} for s in range(5)}
PE = {r: np.mean([P[s][r] for s in range(5)], 0) for r in ROLES}
pred_s = {r: P[0][r].argmax(1) for r in ROLES}
pred_e = {r: PE[r].argmax(1) for r in ROLES}
err_s = {r: (pred_s[r] != label[r]).astype(float) for r in ROLES}
err_e = {r: (pred_e[r] != label[r]).astype(float) for r in ROLES}


def auc_pt(u, err):
    return Scorer(u, err).auc(np.ones(len(u)))


# ---- k selection for kNN on val_id only (per seed, by error-detection AUROC of that seed's own errors)
def choose_k(seed):
    errv = (P[seed]["val_id"].argmax(1) != label["val_id"]).astype(float)
    a = {k: auc_pt(sc[f"s{seed}/val_id/knn{k}"], errv) for k in KS}
    return max(a, key=a.get), a


K_SEL = {s: choose_k(s) for s in range(5)}

# ---- metadata detector (control): inference-time metadata only, fit on val_id
rows = {r: rd(f"S1_tr2013_taskB_{r}.jsonl.gz") for r in ROLES}
for r in ROLES:
    assert [x["instance_id"] for x in rows[r]] == list(T(r, "instance_id"))
_R = roles_from_outfits("S1_tr2013_outfits.jsonl.gz")
_cc, _t1 = collections.Counter(), collections.Counter()
for o in _R["train"]:
    for i in range(off[o], off[o + 1]):
        _cc[(int(cat1[i]), int(cat2[i]))] += 1; _t1[int(cat1[i])] += 1


def meta_X(r, pred):
    X = []
    for x, j in zip(rows[r], pred):
        cp = [np.log1p(item_price[i]) for i in x["ctx"] if item_price[i] > 0]
        med = float(np.median(cp)) if cp else 0.0
        lp = [np.log1p(p) if p > 0 else np.nan for p in x["cand_price"]]
        avail = [v for v in lp if not np.isnan(v)]
        pj = lp[j]; miss = float(np.isnan(pj))
        dist = [abs(v - med) if not np.isnan(v) else np.nan for v in lp]
        rank = (np.sum(np.array([d for d in dist if not np.isnan(d)]) < dist[j]) / 3.0) if not np.isnan(dist[j]) else 0.0
        oh = np.zeros(7); oh[x["cat1"] - 10] = 1
        X.append(np.r_[len(x["ctx"]), oh, med, float(bool(cp)), 0.0 if miss else pj, miss, 0.0 if miss else dist[j],
                       (max(avail) - min(avail)) if len(avail) > 1 else 0.0,
                       np.log((_cc[(x["cat1"], x["cand_cat2"][j])] + 1) / (_t1[x["cat1"]] + 1)), rank])
    return np.array(X, float)


def meta_scores(pred, err):
    Xv = meta_X("val_id", pred["val_id"]); mu, sd = Xv.mean(0), Xv.std(0) + 1e-6
    Z = (Xv - mu) / sd; yv = err["val_id"]; oof = np.zeros(len(yv))
    for tr, te in GroupKFold(5).split(Z, yv, users["val_id"]):
        oof[te] = LogisticRegression(C=1.0, max_iter=500).fit(Z[tr], yv[tr]).decision_function(Z[te])
    clf = LogisticRegression(C=1.0, max_iter=500).fit(Z, yv)
    out = {"val_id": oof}
    for r in EVAL:
        out[r] = clf.decision_function((meta_X(r, pred[r]) - mu) / sd)
    return out


META_S = meta_scores(pred_s, err_s)
META_E = meta_scores(pred_e, err_e)


def build(role):
    k0 = K_SEL[0][0]
    knn = sc[f"s0/{role}/knn{k0}"]
    Ps, Pe = P[0][role], PE[role]
    h_e = entropy(Pe); mi = h_e - np.mean([entropy(P[s][role]) for s in range(5)], 0)
    peers = 1 - np.mean([(P[s][role].argmax(1) == pred_s[role]) for s in range(1, 5)], 0)
    vote = 1 - np.mean([(P[s][role].argmax(1) == pred_e[role]) for s in range(5)], 0)
    S = {"msp_single": -Ps.max(1), "knn_pair": knn, "ens_msp": -Pe.max(1), "ens_entropy": h_e, "ens_mi": mi, "ens_vote_peers": peers, "meta": META_S[role]}
    E = {"ens_msp": -Pe.max(1), "ens_entropy": h_e, "ens_mi": mi, "ens_vote": vote, "msp_single": -Ps.max(1), "knn_pair": knn, "meta": META_E[role]}
    return S, E


SC = {r: build(r) for r in ROLES}
FAM = {"S": dict(err=err_s, idx=0, ref="msp_single", pred=pred_s), "E": dict(err=err_e, idx=1, ref="ens_msp", pred=pred_e)}


def tau_val(fam, name):
    u = SC["val_id"][FAM[fam]["idx"]][name]
    return np.sort(u)[int(np.ceil(0.8 * len(u))) - 1]


TAU = {(f, n): tau_val(f, n) for f in FAM for n in SC["val_id"][FAM[f]["idx"]]}


def analyze(fam, role, B=2000, seed=0, mask=None, methods=None):
    """paired user-cluster bootstrap for every method of the family on one set (optionally a subset)."""
    f = FAM[fam]; U = SC[role][f["idx"]]; err = f["err"][role]; us = users[role]
    m = np.ones(len(err), bool) if mask is None else mask
    W = boot_weights(us[m], B, seed if seed else seed_of(role, fam, int(m.sum()))); ones = np.ones(m.sum())
    res = {}
    for name in (methods or U):
        u = U[name][m]; S = Scorer(u, err[m]); acc = (u <= TAU[(fam, name)]).astype(float)
        pt = dict(auc=S.auc(ones), risk80=S.risk_cov(ones)); pt["cov"], pt["risk_fix"], pt["hce_all"] = S.fixed(ones, acc)
        bs = {k: np.empty(B) for k in pt}
        for b in range(B):
            w = W[b].astype(float); bs["auc"][b] = S.auc(w); bs["risk80"][b] = S.risk_cov(w)
            bs["cov"][b], bs["risk_fix"][b], bs["hce_all"][b] = S.fixed(w, acc)
        res[name] = (pt, bs)
    base = dict(err=float(err[m].mean())); bb = (W * err[m]).sum(1) / W.sum(1)
    return res, base, bb, int(m.sum()), len(set(us[m]))
