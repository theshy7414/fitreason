"""Follow-up diagnostics raised by checks 1-3 (frozen manifests, nothing trained):
 (a) train co-occurrence route: does the positive co-occur with a query item in a TRAIN outfit more than negatives do?
     + a pure id-level 'train co-occurrence count' baseline for Task B.  Also the partial-overlap (>=50% items) outfits flagged in check 1.
 (b) is the training-free cosine drop under shift caused by centring with the 2013 mean? (year-matched centring = transductive diagnostic only)
Run: PYTHONPATH=./pylibs:./scripts python3 scripts/44_followup_diagnostics.py | tee results/exp1_check4_followup.txt
"""
import json, collections, itertools
import numpy as np
from exp1_nn import vec
from exp1_common import *
from importlib import import_module

roles1 = roles_from_outfits("S1_tr2013_outfits.jsonl.gz")
train1 = roles1["train"]
roles = ["val_id", "test_id", "test_ud", "test_2014", "test_2015", "test_2017"]
flags = json.load(open(os.path.join(RES, "exp1_neardup_flags_taskB.json")))

# ---- train pair counts and item -> train outfits
pair = collections.Counter(); t_items = collections.defaultdict(set)
for o in train1:
    its = sorted(set(items_of(o).tolist()))
    for it in its:
        t_items[it].add(int(o))
    for a, b in itertools.combinations(its, 2):
        pair[(a, b)] += 1
print(f"train outfits={len(train1):,} distinct train item pairs={len(pair):,}")

H("(a) train co-occurrence route (id-level), Task B frozen instances")
print(f"{'role':<10}{'n':>6}{'pos co-occurs w/ query item':>30}{'neg co-occurs':>16}{'decidable%':>12}{'cooc-count baseline acc':>26}{'partial-overlap(>=50%) n / pos in that train outfit':>54}")
for role in roles:
    rows = rd(f"S1_tr2013_taskB_{role}.jsonl.gz"); users = np.array([r["user_id"] for r in rows])
    pp, pn, dec, acc, pov, pin = [], [], [], [], 0, 0
    for r in rows:
        ctx = r["ctx"]; k = r["label"]
        cnt = np.array([sum(pair.get((min(c, q), max(c, q)), 0) for q in ctx) for c in r["cands"]], dtype=float)
        has = cnt > 0
        negm = np.arange(4) != k
        pp.append(has[k]); pn.append(has[negm].mean()); dec.append(has[k] and not has[negm].any())
        acc.append(tie_acc(cnt, k))
        if flags[r["instance_id"]]["id_leak50"]:
            pov += 1
            qs = set(ctx) | {r["cands"][k]}
            shared = [to for to in set().union(*[t_items.get(i, set()) for i in ctx]) if len(qs & set(items_of(to).tolist())) >= 0.5 * len(qs)]
            pin += any(r["cands"][k] in set(items_of(to).tolist()) for to in shared)
    print(f"{role:<10}{len(rows):>6}{np.mean(pp)*100:>29.1f}%{np.mean(pn)*100:>15.1f}%{np.mean(dec)*100:>11.1f}%{fmt(cluster_boot(np.array(acc), users)):>26}{f'{pov} / {pin}':>54}")
print("decidable% = positive co-occurs with a query item in some train outfit and NO negative does (a memorised-pair signal exists).")

H("(b) cosine drop under shift: centring with 2013-train mean vs year-matched mean (transductive diagnostic, test-year items' own mean)")
T = items_in(train1); mu = vec(T).astype(np.float32).mean(0)


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


print(f"{'role':<10}{'cos centred on 2013 train mean':>34}{'cos centred on own-set mean':>30}{'cos raw':>20}")
for role in roles:
    rows = rd(f"S1_tr2013_taskB_{role}.jsonl.gz"); users = np.array([r["user_id"] for r in rows])
    mu_own = vec(items_in(roles1[role])).astype(np.float32).mean(0)
    res = {"tr": [], "own": [], "raw": []}
    for r in rows:
        allit = np.array(r["ctx"] + r["cands"]); V = vec(allit).astype(np.float32); nq = len(r["ctx"])
        for key, m in (("tr", mu), ("own", mu_own), ("raw", 0.0)):
            X = unit(V - m); q = unit(X[:nq].mean(0)); res[key].append(tie_acc(X[nq:] @ q, r["label"]))
    print(f"{role:<10}{fmt(cluster_boot(np.array(res['tr']), users)):>34}{fmt(cluster_boot(np.array(res['own']), users)):>30}{fmt(cluster_boot(np.array(res['raw']), users)):>20}")
