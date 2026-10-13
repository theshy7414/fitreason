"""Experiment 1 preflight: audit frozen manifests (leakage assertions, user/item overlap, effective users) and run
METADATA-ONLY shortcut baselines on the frozen Task A / Task B evaluation instances. No image features, no training.
Run: PYTHONPATH=./pylibs python3 scripts/21_audit_and_baselines.py | tee results/exp1_manifest_report.txt
"""
import os, json, gzip, hashlib, collections
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAN = os.path.join(ROOT, "manifests")
Z = np.load(os.path.join(ROOT, "data", "outfits", "outfits_compact.npz"))
set_id, user_id, n_items, year = (Z[k] for k in ["set_id", "user_id", "n_items", "year"])
item_id, cat1, cat2, price = (Z[k] for k in ["item_id", "cat1", "cat2", "price"])
off = np.zeros(len(set_id) + 1, dtype=np.int64); np.cumsum(n_items, out=off[1:])
uniq, first = np.unique(item_id, return_index=True)
item_cat1 = dict(zip(uniq.tolist(), cat1[first].tolist()))
item_cat2 = dict(zip(uniq.tolist(), cat2[first].tolist()))
item_price = dict(zip(uniq.tolist(), price[first].tolist()))


def rd(name):
    with gzip.open(os.path.join(MAN, name), "rt") as f:
        return [json.loads(l) for l in f]


def H(t):
    print("\n" + "=" * 8, t, "=" * 8, flush=True)


def sig(o):
    return hashlib.md5(np.sort(item_id[off[o]:off[o + 1]]).astype(np.int64).tobytes()).hexdigest()[:16]


def eff_users(users):
    c = np.array(list(collections.Counter(users).values()), dtype=float)
    return (c.sum() ** 2) / (c ** 2).sum(), len(c), np.sort(c)[::-1][:10].sum() / c.sum()


def cluster_boot(vals, users, B=1000, seed=0):
    """user-cluster bootstrap CI of the mean of per-instance values."""
    u, inv = np.unique(users, return_inverse=True)
    s = np.bincount(inv, weights=vals); n = np.bincount(inv).astype(float)
    rs = np.random.RandomState(seed)
    m = np.empty(B)
    for b in range(B):
        i = rs.randint(len(u), size=len(u)); m[b] = s[i].sum() / n[i].sum()
    return vals.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def fmt(x):
    m, lo, hi = x
    return f"{m*100:5.1f} [{lo*100:5.1f},{hi*100:5.1f}]"


def tie_acc(scores, label):  # expected accuracy with random tie-break
    scores = np.asarray(scores, dtype=float)
    best = scores == scores.max()
    return (1.0 / best.sum()) if best[label] else 0.0


# =====================================================================================
H("A. TASK B (project split S1): audit")
for tag, tests in [("tr2013", ["val_id", "test_id", "test_ud", "test_2014", "test_2015", "test_2017"]),
                   ("tr2014", ["val_id", "test_id", "test_ud", "test_2015", "test_2016", "test_2017"])]:
    outf = rd(f"S1_{tag}_outfits.jsonl.gz")
    byrole = collections.defaultdict(list)
    for r in outf:
        byrole[r["role"]].append(r)
    train_o = [r["o"] for r in byrole["train"]]
    train_sets = {r["set_id"] for r in byrole["train"]}
    train_users = {r["user_id"] for r in byrole["train"]}
    train_sigs = {sig(o) for o in train_o}
    # outfit-level disjointness + exact-duplicate audit
    allsets = collections.Counter(r["set_id"] for r in outf)
    assert max(allsets.values()) == 1, "an outfit appears in two roles"
    ev_sigs = {role: {sig(r["o"]) for r in rs} for role, rs in byrole.items() if role != "train"}
    print(f"[{tag}] train outfits={len(train_o):,} users={len(train_users):,}; eval roles disjoint from train/each other: OK")
    for role, ss in ev_sigs.items():
        assert not (ss & train_sigs), f"exact duplicate item-set between train and {role}"
    print(f"[{tag}] exact item-set duplicates between train and any eval set: 0 (assert passed)")
    train_items = set(); freq = collections.Counter()
    for o in train_o:
        for i in item_id[off[o]:off[o + 1]].tolist():
            freq[i] += 1
    train_items = set(freq)
    print(f"[{tag}] train unique items={len(train_items):,}")
    # user-held-out audit
    ud_users = {r["user_id"] for r in byrole["test_ud"]}
    assert not (ud_users & train_users), "UD test users leak into train"
    print(f"[{tag}] test_ud users={len(ud_users):,}, overlap with train users = 0 (assert passed)")
    print(f"\n{'role':<10}{'n_inst':>7}{'users':>7}{'eff.users':>10}{'top10u%':>8}{'user_seen%':>11}{'pos_seen%':>10}{'neg_seen%':>10}{'ctx_len(med)':>13}{'pos@0..3 %':>20}")
    for role in tests:
        rows = rd(f"S1_{tag}_taskB_{role}.jsonl.gz")
        for r in rows:  # hard leakage assertions
            pos = r["cands"][r["label"]]
            assert pos not in r["ctx"], "masked item remains in context"
            assert len(set(r["cands"])) == 4
            assert not (set(r["cands"]) - {pos}) & set(r["ctx"]), "negative inside the query outfit"
            assert all(item_cat1[c] == r["cat1"] for c in r["cands"]), "negative has different cat1"
            assert r["set_id"] not in train_sets
        ee, nu, t10 = eff_users([r["user_id"] for r in rows])
        pos_hist = np.bincount([r["label"] for r in rows], minlength=4) / len(rows)
        print(f"{role:<10}{len(rows):>7}{nu:>7}{ee:>10.0f}{t10*100:>8.1f}{np.mean([r['user_seen'] for r in rows])*100:>11.1f}"
              f"{np.mean([r['pos_seen'] for r in rows])*100:>10.1f}{np.mean([np.mean(r['neg_seen']) for r in rows])*100:>10.1f}"
              f"{np.median([len(r['ctx']) for r in rows]):>13.0f}   {' '.join(f'{p*100:4.1f}' for p in pos_hist)}")
    print(f"[{tag}] leakage assertions on every instance (masked item not in ctx; 4 distinct candidates; negatives not in query outfit; "
          f"same cat1; query outfit not in train): passed")

# =====================================================================================
H("B. TASK B shortcut baselines (metadata only; train-year statistics; random tie-break; acc % [user-cluster 95% CI]; chance=25.0)")
tag = "tr2013"
outf = rd(f"S1_{tag}_outfits.jsonl.gz")
train_o = [r["o"] for r in outf if r["role"] == "train"]
cc = collections.Counter()
for o in train_o:
    for i in range(off[o], off[o + 1]):
        cc[(int(cat1[i]), int(cat2[i]))] += 1
tot1 = collections.Counter()
for (a, b), v in cc.items():
    tot1[a] += v
cols = ["val_id", "test_id", "test_ud", "test_2014", "test_2015", "test_2017"]
print(f"{'role':<10}{'cat1-only':>20}{'cat2 prior':>20}{'price-near-ctx':>20}{'train popularity':>20}")
for role in cols:
    rows = rd(f"S1_{tag}_taskB_{role}.jsonl.gz")
    users = np.array([r["user_id"] for r in rows])
    a_c1, a_c2, a_pr, a_pop = [], [], [], []
    for k, r in enumerate(rows):
        a_c1.append(tie_acc(np.zeros(4), r["label"]))
        a_c2.append(tie_acc([np.log((cc[(r["cat1"], c2)] + 1) / (tot1[r["cat1"]] + 1)) for c2 in r["cand_cat2"]], r["label"]))
        cp = [item_price[i] for i in r["ctx"] if item_price[i] > 0]
        if cp and all(p > 0 for p in r["cand_price"]):
            med = np.median(np.log1p(cp))
            a_pr.append(tie_acc([-abs(np.log1p(p) - med) for p in r["cand_price"]], r["label"]))
        else:
            a_pr.append(0.25)
        a_pop.append(tie_acc(r["cand_train_freq"], r["label"]))
    print(f"{role:<10}{fmt(cluster_boot(np.array(a_c1), users)):>20}{fmt(cluster_boot(np.array(a_c2), users)):>20}"
          f"{fmt(cluster_boot(np.array(a_pr), users)):>20}{fmt(cluster_boot(np.array(a_pop), users)):>20}")
print("(price baseline falls back to chance when any price is missing/<=0)")

# =====================================================================================
H("C. TASK A (official procedure S0, seed 0): audit + shortcut baselines on the FROZEN instances")
outf = rd("S0_official_tr2013_outfits.jsonl.gz")
train_o = [r["o"] for r in outf if r["role"] == "train"]
train_sigs = {sig(o) for o in train_o}
train_sets = {int(set_id[o]) for o in train_o}
for ye in [2013, 2014, 2015, 2017]:
    te = [r for r in outf if r["role"] == f"test_{ye}"]
    va = [r for r in outf if r["role"] == f"val_{ye}"]
    assert not ({r["set_id"] for r in te} & train_sets) and not ({r["set_id"] for r in va} & train_sets)
    assert not ({r["set_id"] for r in te} & {r["set_id"] for r in va})
    d = np.mean([sig(r["o"]) in train_sigs for r in te])
    print(f"year {ye}: test outfits={len(te)} val={len(va)} | outfit-level overlap with train/val: 0 (assert) | exact item-set duplicates vs train (official does not dedupe): {d*100:.2f}%")
print(f"\n{'year/n_cands':<14}{'chance':>8}{'label==idx0':>13}{'cat-complement':>22}{'size-only':>22}{'size+category':>22}")
for ye in [2013, 2014, 2015, 2017]:
    for nc in (4, 8):
        rows = rd(f"S0_official_tr2013_taskA_nc{nc}_test{ye}.jsonl.gz")
        users = np.array([r["user_id"] for r in rows])
        a_cat, a_size, a_both = [], [], []
        for r in rows:
            xcat = {item_cat1[i] for i in r["query"]}
            ov = np.array([len(xcat & {item_cat1[i] for i in a}) for a in r["answers"]], dtype=float)
            sz = np.array([len(a) for a in r["answers"]])
            a_cat.append(tie_acc(-ov, r["label"]))
            ok = (sz == len(r["query"])) | (sz == len(r["query"]) - 1)
            a_size.append(tie_acc(ok.astype(float), r["label"]) if ok.any() else 1.0 / nc)
            sc = ok * 100.0 - ov
            a_both.append(tie_acc(sc, r["label"]))
        idx0 = np.mean([r["label"] == 0 for r in rows])
        print(f"{ye}/{nc:<10}{100/nc:>8.1f}{idx0*100:>13.1f}{fmt(cluster_boot(np.array(a_cat), users)):>22}{fmt(cluster_boot(np.array(a_size), users)):>22}{fmt(cluster_boot(np.array(a_both), users)):>22}")
print("(official instances place the correct answer at index 0; frozen files here shuffle positions with a fixed seed and store the label. "
      "label==idx0 column should be ~1/n_cands.)")

# =====================================================================================
H("D. REQUIRED ITEM SET")
S = json.load(open(os.path.join(MAN, "manifest_summary.json")))
for k in ["required_S1_tr2013", "required_S0", "required_S1_tr2014", "required_items_total"]:
    print(f"{k}: {S[k]:,}")
print(f"required_items_total fp16 = {S['fp16_GB']:.2f} GB, fp32 = {S['fp32_GB']:.2f} GB; all 2,335,598 catalog items fp16 = {2335598*4096*2/1e9:.1f} GB")
print("duplicates dropped from project pools:", S["dedupe"])
