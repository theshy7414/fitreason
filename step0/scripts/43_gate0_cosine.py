"""Check 3 - Gate 0 cosine baseline on the FROZEN Task A / Task B instances, next to the metadata baselines. Nothing is trained.
Cosine is on VGG16 fc6 after subtracting the TRAIN-item mean (centred) and L2-normalising each item; 'raw' = no centring.
Primary direction (pre-declared in the spec): HIGHEST similarity wins. Reversed/other variants are diagnostics.
Run: PYTHONPATH=./pylibs:./scripts python3 scripts/43_gate0_cosine.py | tee results/exp1_check3_gate0_cosine.txt
"""
import json, collections
import numpy as np
from exp1_nn import vec
from exp1_common import *


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def cos_scores(cands_items, ctx_items, mu):
    """returns dict of per-candidate scores for several cosine variants. cands_items: list of arrays (set or single item)."""
    out = {}
    flat_c = [np.atleast_1d(c) for c in cands_items]
    allit = np.concatenate([np.atleast_1d(ctx_items)] + flat_c)
    V = vec(allit).astype(np.float32)
    for name, m in (("cen", mu), ("raw", 0.0)):
        X = unit(V - m)
        Q = X[:len(np.atleast_1d(ctx_items))]; p = len(Q); q = unit(Q.mean(0))
        sc, mx = [], []
        for c in flat_c:
            C = X[p:p + len(c)]; p += len(c)
            sc.append(float(unit(C.mean(0)) @ q)); mx.append(float((C @ Q.T).max()))
        out[name] = np.array(sc); out[name + "_max"] = np.array(mx)
    return out


# ============================================================================ Task B
H("TASK B (frozen S1 train-2013 instances; chance 25.0). acc % [user-cluster 95% CI]")
roles1 = roles_from_outfits("S1_tr2013_outfits.jsonl.gz")
train1 = roles1["train"]
T = items_in(train1); mu = vec(T).astype(np.float32).mean(0)
cc = collections.Counter(); tot1 = collections.Counter()
for o in train1:
    for i in range(off[o], off[o + 1]):
        cc[(int(cat1[i]), int(cat2[i]))] += 1; tot1[int(cat1[i])] += 1
flags = json.load(open(os.path.join(RES, "exp1_neardup_flags_taskB.json")))
roles = ["val_id", "test_id", "test_ud", "test_2014", "test_2015", "test_2017"]
rowsB = {}
cols = ["cos_cen", "cos_raw", "cos_maxctx", "cos_REV", "cat2prior", "price", "popularity"]
print(f"{'role':<10}" + "".join(f"{c:>20}" for c in cols))
B_acc = {}
for role in roles:
    rows = rd(f"S1_tr2013_taskB_{role}.jsonl.gz")
    A = {c: [] for c in cols}; users = np.array([r["user_id"] for r in rows])
    for r in rows:
        s = cos_scores([np.array([c]) for c in r["cands"]], np.array(r["ctx"]), mu)
        A["cos_cen"].append(tie_acc(s["cen"], r["label"])); A["cos_raw"].append(tie_acc(s["raw"], r["label"]))
        A["cos_maxctx"].append(tie_acc(s["cen_max"], r["label"])); A["cos_REV"].append(tie_acc(-s["cen"], r["label"]))
        A["cat2prior"].append(tie_acc([np.log((cc[(r["cat1"], c2)] + 1) / (tot1[r["cat1"]] + 1)) for c2 in r["cand_cat2"]], r["label"]))
        cp = [item_price[i] for i in r["ctx"] if item_price[i] > 0]
        if cp and all(p > 0 for p in r["cand_price"]):
            med = np.median(np.log1p(cp)); A["price"].append(tie_acc([-abs(np.log1p(p) - med) for p in r["cand_price"]], r["label"]))
        else:
            A["price"].append(0.25)
        A["popularity"].append(tie_acc(r["cand_train_freq"], r["label"]))
    rowsB[role] = (rows, {k: np.array(v) for k, v in A.items()}, users)
    B_acc[role] = {k: cluster_boot(np.array(v), users) for k, v in A.items()}
    print(f"{role:<10}" + "".join(f"{fmt(B_acc[role][c]):>20}" for c in cols))
print("cos_cen=centred mean-query cosine (primary); cos_raw=uncentred; cos_maxctx=max over query items; cos_REV=lowest similarity wins (diagnostic)")

H("TASK B: cosine (centred, primary) by subgroup")
print(f"{'role':<10}{'all':>20}{'pos_seen=T':>20}{'pos_seen=F':>20}{'excl. dup-flagged':>20}{'n flagged':>10}")
for role in roles:
    rows, A, users = rowsB[role]
    ps = np.array([r["pos_seen"] for r in rows])
    bad = np.array([flags[r["instance_id"]]["neg_dup_pos095"] or flags[r["instance_id"]]["cand_query_dup098"] or flags[r["instance_id"]]["outfit_leak095"] for r in rows])
    a = A["cos_cen"]
    f = lambda m: fmt(cluster_boot(a[m], users[m])) if m.sum() > 50 and len(set(users[m])) > 20 else "      -"
    print(f"{role:<10}{fmt(B_acc[role]['cos_cen']):>20}{f(ps):>20}{f(~ps):>20}{f(~bad):>20}{int(bad.sum()):>10}")

# ============================================================================ Task A
H("TASK A (frozen S0 official-procedure instances). acc % [user-cluster 95% CI]")
roles0 = roles_from_outfits("S0_official_tr2013_outfits.jsonl.gz")
T0 = items_in(roles0["train"]); mu0 = vec(T0).astype(np.float32).mean(0)
colsA = ["cos_cen", "cos_raw", "cos_REV", "cat_compl", "size", "size+cat"]
accA = {}
for nc in (4, 8):
    print(f"\n n_cands={nc} (chance {100/nc:.1f})\n{'year':<8}" + "".join(f"{c:>20}" for c in colsA))
    for ye in (2013, 2014, 2015, 2017):
        rows = rd(f"S0_official_tr2013_taskA_nc{nc}_test{ye}.jsonl.gz")
        A = {c: [] for c in colsA}; users = np.array([r["user_id"] for r in rows])
        for r in rows:
            s = cos_scores([np.array(a) for a in r["answers"]], np.array(r["query"]), mu0)
            A["cos_cen"].append(tie_acc(s["cen"], r["label"])); A["cos_raw"].append(tie_acc(s["raw"], r["label"])); A["cos_REV"].append(tie_acc(-s["cen"], r["label"]))
            xcat = {item_cat1[i] for i in r["query"]}
            ov = np.array([len(xcat & {item_cat1[i] for i in a}) for a in r["answers"]], float)
            sz = np.array([len(a) for a in r["answers"]]); ok = (sz == len(r["query"])) | (sz == len(r["query"]) - 1)
            A["cat_compl"].append(tie_acc(-ov, r["label"]))
            A["size"].append(tie_acc(ok.astype(float), r["label"]) if ok.any() else 1.0 / nc)
            A["size+cat"].append(tie_acc(ok * 100.0 - ov, r["label"]))
        accA[(nc, ye)] = {k: cluster_boot(np.array(v), users) for k, v in A.items()}
        print(f"{ye:<8}" + "".join(f"{fmt(accA[(nc, ye)][c]):>20}" for c in colsA))

# ============================================================================ Gate 0 read-out
H("GATE 0 read-out (cosine part)")
meta_up = max(B_acc["test_id"][k][2] for k in ("cat2prior", "price", "popularity"))
best_name = max(("cos_cen", "cos_raw", "cos_maxctx"), key=lambda k: B_acc["test_id"][k][0])
bm, blo, bhi = B_acc["test_id"][best_name]
print(f"Task B test_id: best metadata baseline upper CI = {meta_up*100:.1f}% (price) | best cosine variant '{best_name}' = {fmt((bm, blo, bhi))}")
print(f"  cosine above chance (CI lower > 25)? {blo > 0.25} | cosine above best metadata upper CI? {blo > meta_up}")
print("  => later neural criterion (spec): neural test_id CI lower bound must exceed", f"{max(meta_up, bhi)*100:.1f}%", "(max of metadata upper CI and cosine upper CI)")
