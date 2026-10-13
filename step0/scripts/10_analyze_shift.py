"""Step 0: metadata-only analysis of SHIFT15M outfits: schema, official-filter counts, composition drift,
leakage risks, and shortcut baselines for the official FITB protocol (no image features used).
Run:  PYTHONPATH=./pylibs python3 scripts/10_analyze_shift.py > results/step0_analysis.txt
"""
import os, sys, time, hashlib
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
Z = np.load(os.path.join(ROOT, "data", "outfits", "outfits_compact.npz"))
set_id, user_id, like, n_items, year, ymd = (Z[k] for k in ["set_id", "user_id", "like", "n_items", "year", "ymd"])
item_id, cat1, cat2, price = (Z[k] for k in ["item_id", "cat1", "cat2", "price"])
N = len(set_id)
off = np.zeros(N + 1, dtype=np.int64)
np.cumsum(n_items, out=off[1:])
assert off[-1] == len(item_id)
outfit_of_slot = np.repeat(np.arange(N), n_items)

def H(t):
    print("\n" + "=" * 8, t, "=" * 8, flush=True)

H("1. BASIC")
print(f"outfits={N:,} item_slots={len(item_id):,} unique_items={len(np.unique(item_id)):,} unique_users={len(np.unique(user_id)):,}")
print("publish_date range:", ymd.min(), ymd.max())
print("n_items per outfit: min/median/mean/max =", n_items.min(), int(np.median(n_items)), round(float(n_items.mean()), 2), n_items.max())
print("category_id1 values:", sorted(np.unique(cat1).tolist()))
print("price<=0 slots:", int((price <= 0).sum()))
print("like_num: min/median/mean/max =", like.min(), int(np.median(like)), round(float(like.mean()), 1), like.max())
yc = pd.Series(year).value_counts().sort_index()
print("outfits per publish year:\n", yc.to_string())

H("2. DUPLICATES")
print("duplicate set_id rows:", N - len(np.unique(set_id)))
# outfit signature = hash of sorted item ids
sig = np.empty(N, dtype=np.int64)
for i in range(N):
    sig[i] = hash(tuple(np.sort(item_id[off[i]:off[i + 1]]).tolist()))
sig_ser = pd.Series(sig)
dup_mask = sig_ser.duplicated(keep=False).values
print(f"outfits whose exact item set also appears in another outfit: {int(dup_mask.sum()):,} ({dup_mask.mean()*100:.2f}%)")
dfs = pd.DataFrame({"sig": sig, "year": year, "user": user_id})
g = dfs.groupby("sig").agg(n=("year", "size"), ny=("year", "nunique"), nu=("user", "nunique"))
g = g[g.n > 1]
print(f"duplicated item-sets: {len(g):,}; spanning >1 year: {int((g.ny > 1).sum()):,}; by >1 distinct user: {int((g.nu > 1).sum()):,}")

H("3. OFFICIAL FILTER (>=4 distinct category_id1 AND like_num>=50) PER YEAR")
bit = (1 << (cat1 - cat1.min()).astype(np.int64))
mask_per_outfit = np.bitwise_or.reduceat(bit, off[:-1])
ncat = np.array([bin(int(m)).count("1") for m in np.unique(mask_per_outfit)])
lut = dict(zip(np.unique(mask_per_outfit).tolist(), ncat.tolist()))
num_categories = np.fromiter((lut[int(m)] for m in mask_per_outfit), dtype=np.int16, count=N)
elig = (num_categories >= 4) & (like >= 50)
rows = []
for y in range(2010, 2021):
    m = year == y
    rows.append((y, int(m.sum()), int((m & (num_categories >= 4)).sum()), int((m & (like >= 50)).sum()), int((m & elig).sum())))
print(pd.DataFrame(rows, columns=["year", "all", ">=4cats", "like>=50", "eligible"]).to_string(index=False))
print("official sizes needed: train=30,816 (from train year), val+test=3,851+3,851 = 7,702 (from test year)")

H("4. PER-YEAR COMPOSITION OF ELIGIBLE OUTFITS (2013-2017)")
Y = list(range(2013, 2018))
cat_levels = sorted(np.unique(cat1).tolist())
slot_elig = elig[outfit_of_slot]
comp = {}
for y in Y:
    m_o = elig & (year == y)
    m_s = m_o[outfit_of_slot]
    cdist = pd.Series(cat1[m_s]).value_counts(normalize=True).reindex(cat_levels, fill_value=0)
    pattern = pd.Series(mask_per_outfit[m_o]).value_counts(normalize=True)
    u = pd.Series(user_id[m_o]).value_counts()
    comp[y] = dict(cdist=cdist, pattern=pattern, users=set(u.index.tolist()))
    top1pct = u.head(max(1, int(len(u) * 0.01))).sum() / u.sum()
    print(f"{y}: outfits={int(m_o.sum()):,} users={len(u):,} outfits/user mean={u.mean():.2f} median={u.median():.0f} "
          f"top1%users_share={top1pct*100:.1f}% top10users_share={u.head(10).sum()/u.sum()*100:.1f}% "
          f"items/outfit={n_items[m_o].mean():.2f} like_med={np.median(like[m_o]):.0f} price_med={np.median(price[m_s][price[m_s]>0]):.0f}")
print("\ncategory_id1 share of item slots, per year (cols=category):")
print(pd.DataFrame({y: comp[y]["cdist"] for y in Y}).T.round(3).to_string())
print("\nTotal-variation distance between years, item-level category mix  (0=same, 1=disjoint):")
tv = pd.DataFrame(index=Y, columns=Y, dtype=float)
for a in Y:
    for b in Y:
        tv.loc[a, b] = 0.5 * np.abs(comp[a]["cdist"].values - comp[b]["cdist"].values).sum()
print(tv.round(3).to_string())
print("\nTV distance, outfit-level category-PATTERN mix (which categories co-occur):")
tvp = pd.DataFrame(index=Y, columns=Y, dtype=float)
for a in Y:
    for b in Y:
        idx = comp[a]["pattern"].index.union(comp[b]["pattern"].index)
        tvp.loc[a, b] = 0.5 * np.abs(comp[a]["pattern"].reindex(idx, fill_value=0).values - comp[b]["pattern"].reindex(idx, fill_value=0).values).sum()
print(tvp.round(3).to_string())

H("5. LEAKAGE ACROSS YEARS (train year 2013 -> test years), eligible pools")
# first year each item ever appears in ANY outfit
uniq_items, inv = np.unique(item_id, return_inverse=True)
first_year = np.full(len(uniq_items), 9999, dtype=np.int16)
np.minimum.at(first_year, inv, year[outfit_of_slot])
slot_first_year = first_year[inv]
cat_year = pd.read_csv(os.path.join(ROOT, "data", "catalog", "item_catalog.txt"), sep=" ", header=None,
                       names=["item_id", "cat1", "cat2", "year"])
cy = dict(zip(cat_year.item_id.values, cat_year.year.values))
print("item_catalog rows:", len(cat_year), "| catalog year range:", cat_year.year.min(), cat_year.year.max())
print("catalog year == first outfit-appearance year for", round(float((np.array([cy.get(int(i), -1) for i in uniq_items[:200000]]) == first_year[:200000]).mean()) * 100, 1), "% of the first 200k items")
ys = 2013
train_pool = elig & (year == ys)
train_users = comp[ys]["users"]
all_train_year_users = set(user_id[year == ys].tolist())
train_item_set = set(item_id[train_pool[outfit_of_slot]].tolist())
train_sigs = set(sig[train_pool].tolist())
for ye in [2013, 2014, 2015, 2016, 2017]:
    tp = elig & (year == ye)
    tp_slots = tp[outfit_of_slot]
    users_t = user_id[tp]
    seen_user = np.isin(users_t, list(all_train_year_users)).mean()
    item_seen = np.isin(item_id[tp_slots], list(train_item_set)).mean()
    born_before = (slot_first_year[tp_slots] <= ys).mean()
    sig_seen = np.isin(sig[tp], list(train_sigs)).mean() if ye != ys else float("nan")
    print(f"train {ys} -> test {ye}: test outfits by users who posted in {ys}: {seen_user*100:.1f}% | "
          f"test item slots already in {ys} eligible outfits: {item_seen*100:.1f}% | "
          f"item first appeared <= {ys}: {born_before*100:.1f}% | test outfits with identical item-set seen in {ys} pool: {sig_seen*100:.2f}%")

H("6. SHORTCUT BASELINES for the official set-FITB protocol (metadata only, simulated like shift15m code)")
def simulate(ye, n_cands, seed=0, n_test=3851):
    rs = np.random.RandomState(seed)
    pool = np.where(elig & (year == ye))[0]
    pool = rs.permutation(pool)[: n_test]
    nP = len(pool)
    X_cats, Y_cats, Y_size, X_size = [], [], [], []
    halves = []
    for o in pool:
        sl = np.arange(off[o], off[o + 1])
        ysz = len(sl) // 2
        perm = rs.permutation(len(sl))
        ymask_idx = perm[:ysz]
        xs = sl[perm[ysz:]]; ys_ = sl[ymask_idx]
        halves.append((xs, ys_))
    acc_cat, acc_size, acc_both, ties_cat = 0.0, 0.0, 0.0, 0.0
    cat_overlap_pos, cat_overlap_neg = [], []
    for i in range(nP):
        xs, y_pos = halves[i]
        others = rs.choice(np.delete(np.arange(nP), i), n_cands - 1, replace=False)
        cands = [y_pos]
        for j in others:
            sl = np.arange(off[pool[j]], off[pool[j]+ 1])
            ysz = len(sl) // 2
            cands.append(sl[rs.permutation(len(sl))[:ysz]])
        xcat = set(cat1[xs].tolist())
        ov = np.array([len(xcat & set(cat1[c].tolist())) for c in cands], dtype=float)
        sz = np.array([len(c) for c in cands])
        cat_overlap_pos.append(ov[0]); cat_overlap_neg.extend(ov[1:].tolist())
        # (a) category-complement heuristic: pick candidate(s) with minimum category overlap with the query
        best = ov == ov.min(); acc_cat += best[0] / best.sum()
        # (b) size heuristic: true |Y| is |X| or |X|-1 by construction
        ok = (sz == len(xs)) | (sz == len(xs) - 1)
        if ok.any():
            acc_size += ok[0] / ok.sum()
        else:
            acc_size += 1.0 / n_cands
        # (c) both
        ok2 = ok & (ov == ov[ok].min()) if ok.any() else best
        acc_both += ok2[0] / ok2.sum()
    return dict(n=nP, chance=1.0 / n_cands, category_complement=acc_cat / nP, size_only=acc_size / nP, size_plus_category=acc_both / nP,
                mean_cat_overlap_correct=float(np.mean(cat_overlap_pos)), mean_cat_overlap_negative=float(np.mean(cat_overlap_neg)))
for ye in [2014, 2016]:
    for nc in [4, 8]:
        print(f"test year {ye}, n_cands={nc}:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in simulate(ye, nc).items()})

H("7. STORAGE ESTIMATES (unique items needed per experiment, official-size sampling, approximate)")
def unique_items_for(ys, ye, seed=0):
    rs = np.random.RandomState(seed)
    tr = rs.permutation(np.where(elig & (year == ys))[0])[:30816]
    te = rs.permutation(np.where(elig & (year == ye))[0])[:7702] if ye != ys else np.array([], dtype=int)
    sel = np.concatenate([tr, te])
    sl = np.concatenate([np.arange(off[o], off[o + 1]) for o in sel])
    return len(np.unique(item_id[sl]))
for ys_, ye_ in [(2013, 2013), (2013, 2014), (2013, 2016), (2013, 2017)]:
    u = unique_items_for(ys_, ye_)
    print(f"{ys_}->{ye_}: unique items={u:,} | fp16 4096-d = {u*4096*2/1e9:.2f} GB | fp32 = {u*4096*4/1e9:.2f} GB")
tot_items = len(uniq_items)
print(f"all {tot_items:,} items: fp16 = {tot_items*4096*2/1e9:.1f} GB | fp32 = {tot_items*4096*4/1e9:.1f} GB")
print("done", flush=True)
