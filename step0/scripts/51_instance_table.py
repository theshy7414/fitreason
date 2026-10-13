"""Per-instance table for Task B (frozen S1 train-2013 manifests): baseline correctness (expected acc with random tie-break) + audit flags.
Used by Phase 1/2 reporting. Nothing is re-sampled. Output: results/exp1_instance_table_taskB.npz  (keys '<role>/<field>')
Run: PYTHONPATH=./pylibs:./scripts python3 scripts/51_instance_table.py
"""
import json, collections, itertools
import numpy as np
from exp1_nn import vec, nn_tables
from exp1_common import *

roles = ["val_id", "test_id", "test_ud", "test_2014", "test_2015", "test_2017"]
R = roles_from_outfits("S1_tr2013_outfits.jsonl.gz")
train1 = R["train"]
evo = np.concatenate([R[r] for r in roles])
T = items_in(train1); mu = vec(T).astype(np.float32).mean(0)
flags = json.load(open(os.path.join(RES, "exp1_neardup_flags_taskB.json")))

# visual twins / NN similarity of every eval item vs train items (same category, identical id excluded)
_, U, _, nn_s, _, _ = nn_tables(train1, evo)
posU = {int(x): k for k, x in enumerate(U)}

# train pair counts, category-2 prior
pair = collections.Counter(); cc = collections.Counter(); tot1 = collections.Counter()
for o in train1:
    its = sorted(set(items_of(o).tolist()))
    for a, b in itertools.combinations(its, 2):
        pair[(a, b)] += 1
    for i in range(off[o], off[o + 1]):
        cc[(int(cat1[i]), int(cat2[i]))] += 1; tot1[int(cat1[i])] += 1


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


out = {}
for role in roles:
    rows = rd(f"S1_tr2013_taskB_{role}.jsonl.gz")
    f = collections.defaultdict(list)
    for r in rows:
        k = r["label"]; ctx = r["ctx"]; cands = r["cands"]
        f["instance_id"].append(r["instance_id"]); f["user_id"].append(r["user_id"]); f["label"].append(k)
        f["cat1"].append(r["cat1"]); f["ctx_len"].append(len(ctx)); f["user_seen"].append(r["user_seen"]); f["pos_idseen"].append(r["pos_seen"])
        V = vec(np.array(ctx + cands)).astype(np.float32); X = unit(V - mu); q = unit(X[:len(ctx)].mean(0))
        f["acc_cosine"].append(tie_acc(X[len(ctx):] @ q, k))
        cp = [item_price[i] for i in ctx if item_price[i] > 0]
        if cp and all(p > 0 for p in r["cand_price"]):
            med = np.median(np.log1p(cp)); f["acc_price"].append(tie_acc([-abs(np.log1p(p) - med) for p in r["cand_price"]], k))
        else:
            f["acc_price"].append(0.25)
        f["acc_cat2prior"].append(tie_acc([np.log((cc[(r["cat1"], c2)] + 1) / (tot1[r["cat1"]] + 1)) for c2 in r["cand_cat2"]], k))
        f["acc_popularity"].append(tie_acc(r["cand_train_freq"], k))
        cnt = np.array([sum(pair.get((min(c, qq), max(c, qq)), 0) for qq in ctx) for c in cands], float)
        f["acc_cooc"].append(tie_acc(cnt, k)); has = cnt > 0
        f["decidable"].append(bool(has[k] and not np.delete(has, k).any()))
        s_pos = nn_s[posU[cands[k]], 0]
        f["pos_nn_sim"].append(float(s_pos)); f["pos_twin"].append(bool(s_pos >= 0.999))
        fl = flags[r["instance_id"]]
        f["id_leak50"].append(fl["id_leak50"])
        f["nd_flagged"].append(bool(fl["neg_dup_pos095"] or fl["cand_query_dup098"] or fl["outfit_leak095"]))
    for k_, v in f.items():
        arr = np.array(v)
        out[f"{role}/{k_}"] = arr.astype(str) if k_ == "instance_id" else arr
    st = np.where(out[f"{role}/pos_idseen"], 0, np.where(out[f"{role}/pos_twin"], 1, 2))
    out[f"{role}/stratum"] = st      # 0 id-seen, 1 twin-only (id unseen, visual twin), 2 visually novel
    print(f"{role}: n={len(rows)} cosine={np.mean(f['acc_cosine'])*100:.1f} decidable={np.mean(f['decidable'])*100:.1f}% "
          f"strata id-seen/twin-only/novel = {[(st==s).mean().round(3) for s in range(3)]}", flush=True)
np.savez(os.path.join(RES, "exp1_instance_table_taskB.npz"), **out)
print("saved results/exp1_instance_table_taskB.npz")
