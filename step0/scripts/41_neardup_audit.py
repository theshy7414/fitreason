"""Check 1 - near-duplicate audit train vs eval on the FROZEN manifests (no re-sampling).
Similarity = cosine of VGG16 fc6 after subtracting the TRAIN-item mean (raw fc6 cosine is dominated by a shared mean vector),
computed WITHIN category_id1 (a re-listed product keeps its category), identical item_id excluded (that is the 'seen item' case,
tracked separately). Run: PYTHONPATH=./pylibs python3 scripts/41_neardup_audit.py | tee results/exp1_check1_neardup.txt
"""
import collections, json
import numpy as np, torch
from exp1_common import *

dev = "cuda"
F, rid = load_required_features()
TAUS = [0.90, 0.95, 0.98, 0.999]


def vec(items):
    return F[np.searchsorted(rid, items)]


def normed(items, mu):
    x = vec(items).astype(np.float32) - mu
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x


def topk_sims(Ev, e_ids, Tt, tid, k=10, chunk=2048):
    outv, outi = [], []
    for s in range(0, len(Ev), chunk):
        E = torch.from_numpy(Ev[s:s + chunk]).to(dev)
        S = (E @ Tt.T).float()
        eid = torch.from_numpy(e_ids[s:s + chunk]).to(dev)
        S[eid[:, None] == tid[None, :]] = -2.0
        v, i = S.topk(min(k, S.shape[1]), dim=1)
        outv.append(v.cpu().numpy()); outi.append(tid[i].cpu().numpy())
    return np.concatenate(outv), np.concatenate(outi)


def nn_tables(train_o, eval_outfits):
    T = items_in(train_o); U = items_in(eval_outfits)
    mu = vec(T).astype(np.float32).mean(0)
    nn_s = np.full((len(U), 10), -2.0, np.float32); nn_i = np.full((len(U), 10), -1, np.int64)
    ref_top1 = collections.defaultdict(list)
    t_cat = np.array([item_cat1[i] for i in T]); u_cat = np.array([item_cat1[i] for i in U])
    rs = np.random.RandomState(0)
    for c in np.unique(u_cat):
        Tc = T[t_cat == c]; Uc_pos = np.where(u_cat == c)[0]
        Tt = torch.from_numpy(normed(Tc, mu)).half().to(dev); tid = torch.from_numpy(Tc).to(dev)
        v, i = topk_sims(normed(U[Uc_pos], mu).astype(np.float16), U[Uc_pos], Tt, tid)
        nn_s[Uc_pos, :v.shape[1]] = v; nn_i[Uc_pos, :v.shape[1]] = i
        sub = rs.choice(len(Tc), min(3000, len(Tc)), replace=False)        # reference: train -> other train items
        v2, _ = topk_sims(normed(Tc[sub], mu).astype(np.float16), Tc[sub], Tt, tid, k=1)
        ref_top1[int(c)] = v2[:, 0]
        del Tt
    return T, U, mu, nn_s, nn_i, ref_top1


def outfit_leak(train_o, outfits, U, nn_s, nn_i, T):
    """per eval outfit: max fraction of its items matched (near-dup, per tau; or identical id) to ONE train outfit."""
    t_items = collections.defaultdict(list)
    for o in train_o:
        for it in items_of(o).tolist():
            t_items[it].append(int(o))
    pos = {int(x): k for k, x in enumerate(U)}
    res = {}
    for o in outfits:
        its = np.unique(items_of(o)).tolist(); n = len(its)
        d = {}
        cnt = collections.defaultdict(set)
        for it in its:                                    # identical-item overlap
            for to in t_items.get(it, []):
                cnt[to].add(it)
        d["id"] = max([len(s) for s in cnt.values()] + [0]) / n
        for tau in TAUS:
            cnt = collections.defaultdict(set)
            for it in its:
                k = pos[it]
                for s, nid in zip(nn_s[k], nn_i[k]):
                    if s >= tau:
                        for to in t_items.get(int(nid), []):
                            cnt[to].add(it)
            d[tau] = max([len(s) for s in cnt.values()] + [0]) / n
        res[int(o)] = d
    return res


def pct(a):
    return " ".join(f"{np.percentile(a, p):.3f}" for p in (5, 25, 50, 75, 95, 99))


def report_items(tag, train_o, roles, T, U, nn_s, ref_top1):
    Tset = set(T.tolist()); pos = {int(x): k for k, x in enumerate(U)}
    allref = np.concatenate(list(ref_top1.values()))
    print(f"[{tag}] reference train->other-train top-1 sim percentiles(5,25,50,75,95,99): {pct(allref)}  "
          f"| share>=" + " ".join(f"{t}:{(allref>=t).mean()*100:.1f}%" for t in TAUS))
    print(f"{'role':<10}{'uniq items':>11}{'seen-id%':>9}{'NN pct(5..99) of UNSEEN items':>40}" + "".join(f"{'>='+str(t):>9}" for t in TAUS))
    for role, O in roles.items():
        its = items_in(O); un = np.array([i for i in its if i not in Tset])
        top1 = nn_s[[pos[int(i)] for i in un], 0]
        print(f"{role:<10}{len(its):>11}{(1-len(un)/len(its))*100:>9.1f}{pct(top1):>40}" + "".join(f"{(top1>=t).mean()*100:>8.2f}%" for t in TAUS))
    print("   (rates are among items whose item_id is NOT in train; identical-id reuse is the 'seen' column)")


def report_outfits(tag, leak, roles):
    print(f"\n[{tag}] OUTFIT-LEVEL: share of eval outfits whose items are matched to ONE train outfit")
    print(f"{'role':<10}{'n':>6}" + "".join(f"{('id>='+str(f)+'/near>=.95>='+str(f)):>0}" for f in []) +
          f"{'id>=.5':>9}{'id>=.8':>8}" + "".join(f"{'nd'+str(t)+'>=.5':>12}{'>=.8':>7}" for t in TAUS[1:]))
    for role, O in roles.items():
        L = [leak[int(o)] for o in O]
        row = f"{role:<10}{len(O):>6}{np.mean([l['id']>=0.5 for l in L])*100:>8.2f}%{np.mean([l['id']>=0.8 for l in L])*100:>7.2f}%"
        for t in TAUS[1:]:
            row += f"{np.mean([l[t]>=0.5 for l in L])*100:>11.2f}%{np.mean([l[t]>=0.8 for l in L])*100:>6.2f}%"
        print(row)


# ============================================================== Task B / S1 (train 2013)
H("S1 train-2013 (Task B): item-level near-duplicates vs train items")
roles1 = roles_from_outfits("S1_tr2013_outfits.jsonl.gz")
train1 = roles1.pop("train")
evo = np.concatenate(list(roles1.values()))
T1, U1, mu1, nn_s1, nn_i1, ref1 = nn_tables(train1, evo)
report_items("S1", train1, roles1, T1, U1, nn_s1, ref1)
print("\nPer-category reference vs shifted-year NN (median top-1 sim of UNSEEN items; lower = more visually novel):")
Tset1 = set(T1.tolist()); pos1 = {int(x): k for k, x in enumerate(U1)}
cats = sorted(set(item_cat1[int(i)] for i in U1))
print(f"{'cat1':<6}{'train-ref':>10}" + "".join(f"{r:>10}" for r in roles1))
for c in cats:
    row = f"{c:<6}{np.median(ref1[c]):>10.3f}"
    for role, O in roles1.items():
        un = [i for i in items_in(O) if item_cat1[int(i)] == c and int(i) not in Tset1]
        row += f"{np.median(nn_s1[[pos1[int(i)] for i in un], 0]):>10.3f}" if un else f"{'-':>10}"
    print(row)

H("S1: outfit-level leakage")
leak1 = outfit_leak(train1, evo, U1, nn_s1, nn_i1, T1)
report_outfits("S1", leak1, roles1)

# ---------------------------------------------------------------- instance-level (Task B)
H("S1: instance-level (Task B frozen instances)")
flags = {}
print(f"{'role':<10}{'n':>6} | P(pos has near-dup in train) vs P(neg has) at tau=.95 | neg~pos dup .95/.98 | cand~query-item dup .95/.98 | outfit-leak(.95,>=.5)")
for role in roles1:
    rows = rd(f"S1_tr2013_taskB_{role}.jsonl.gz")
    p_pos, p_neg, npd95, npd98, cq95, cq98, lk = [], [], [], [], [], [], []
    for r in rows:
        cands = np.array(r["cands"]); k = r["label"]
        s_top = np.array([nn_s1[pos1[int(c)], 0] for c in cands])
        negm = np.arange(4) != k
        p_pos.append(s_top[k] >= 0.95); p_neg.append((s_top[negm] >= 0.95).mean())
        X = normed(np.concatenate([cands, np.array(r["ctx"])]), mu1)
        C, Q = X[:4], X[4:]
        cc = C @ C[k]; npd95.append(bool((cc[negm] >= 0.95).any())); npd98.append(bool((cc[negm] >= 0.98).any()))
        cq = (C @ Q.T).max(1)
        cq95.append(bool((cq >= 0.95).any())); cq98.append(bool((cq >= 0.98).any()))
        L = leak1[next(o for o in roles1[role] if int(set_id[o]) == r["set_id"])]
        lk.append(L[0.95] >= 0.5)
        flags[r["instance_id"]] = dict(role=role, outfit_leak095=bool(L[0.95] >= 0.5), outfit_leak098=bool(L[0.98] >= 0.5), id_leak50=bool(L["id"] >= 0.5),
                                       neg_dup_pos095=bool(npd95[-1]), neg_dup_pos098=bool(npd98[-1]), cand_query_dup098=bool(cq98[-1]),
                                       pos_nd095=bool(s_top[k] >= 0.95))
    print(f"{role:<10}{len(rows):>6} | {np.mean(p_pos)*100:5.1f}% vs {np.mean(p_neg)*100:5.1f}% | {np.mean(npd95)*100:5.2f}% / {np.mean(npd98)*100:5.2f}% "
          f"| {np.mean(cq95)*100:5.2f}% / {np.mean(cq98)*100:5.2f}% | {np.mean(lk)*100:5.2f}%")
json.dump(flags, open(os.path.join(RES, "exp1_neardup_flags_taskB.json"), "w"))

# ============================================================== Task A / S0 (official procedure)
H("S0 official-procedure split (Task A): near-duplicates vs S0 train items")
roles0 = roles_from_outfits("S0_official_tr2013_outfits.jsonl.gz")
train0 = roles0.pop("train")
roles0 = {k: v for k, v in roles0.items() if k.startswith("test_")}
ev0 = np.concatenate(list(roles0.values()))
T0, U0, mu0, nn_s0, nn_i0, ref0 = nn_tables(train0, ev0)
report_items("S0", train0, roles0, T0, U0, nn_s0, ref0)
leak0 = outfit_leak(train0, ev0, U0, nn_s0, nn_i0, T0)
report_outfits("S0", leak0, roles0)
json.dump({int(set_id[o]): {str(k): v for k, v in d.items()} for o, d in leak0.items()}, open(os.path.join(RES, "exp1_neardup_flags_taskA_outfits.json"), "w"))
print("\nflags saved: results/exp1_neardup_flags_taskB.json, results/exp1_neardup_flags_taskA_outfits.json (manifests untouched)")
