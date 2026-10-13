"""Step 0: how much of the apparent year-to-year drift in outfit composition is BETWEEN-user (who is posting)
versus WITHIN-user (the same people changing what they post)?
Run: PYTHONPATH=./pylibs python3 scripts/13_user_confound_check.py > results/step0_user_confound.txt
Metric: total-variation (TV) distance between distributions of the 'category pattern' of an outfit
(bitmask of which of the 7 category_id1 values are present), 0 = identical, 1 = disjoint.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
Z = np.load(os.path.join(ROOT, "data", "outfits", "outfits_compact.npz"))
year, like, n_items, user, cat1 = Z["year"], Z["like"], Z["n_items"], Z["user_id"], Z["cat1"]
N = len(year)
off = np.zeros(N + 1, dtype=np.int64); np.cumsum(n_items, out=off[1:])
bit = 1 << (cat1 - cat1.min()).astype(np.int64)
mask = np.bitwise_or.reduceat(bit, off[:-1])
popc = np.array([bin(i).count("1") for i in range(128)])
ncat = popc[mask]
df = pd.DataFrame({"user": user, "year": year, "mask": mask, "like": like, "ncat": ncat})

def tv(p, q):
    idx = p.index.union(q.index)
    return 0.5 * np.abs(p.reindex(idx, fill_value=0).values - q.reindex(idx, fill_value=0).values).sum()

def user_equal_weight_dist(d):
    """mean over users of each user's own pattern distribution (every user counts equally)."""
    c = d.groupby(["user", "mask"]).size().rename("n").reset_index()
    c["p"] = c["n"] / c.groupby("user")["n"].transform("sum")
    return c.groupby("mask")["p"].sum() / d["user"].nunique()

def run(name, sub):
    print(f"\n##### filter = {name}  (outfits={len(sub):,})")
    print("year | outfits | users | outfits/user mean | top-10 users share | top-1% users share")
    for y in range(2013, 2018):
        s = sub[sub.year == y]
        u = s.user.value_counts()
        print(f"{y} | {len(s):,} | {len(u):,} | {u.mean():.1f} | {u.head(10).sum()/u.sum()*100:.1f}% | {u.head(max(1,int(len(u)*0.01))).sum()/u.sum()*100:.1f}%")
    print("\npair | users active in both | overall TV (all outfits) | within-user TV (same users, equal weights) | same-year split-half noise floor (same users)")
    rs = np.random.RandomState(0)
    for a, b in [(2013, 2014), (2013, 2015), (2013, 2016), (2013, 2017), (2015, 2016)]:
        sa, sb = sub[sub.year == a], sub[sub.year == b]
        both = np.intersect1d(sa.user.unique(), sb.user.unique())
        overall = tv(sa["mask"].value_counts(normalize=True), sb["mask"].value_counts(normalize=True))
        if len(both) < 20:
            print(f"{a}->{b} | {len(both)} | {overall:.3f} | n/a (too few shared users) | n/a")
            continue
        ua, ub = sa[sa.user.isin(both)], sb[sb.user.isin(both)]
        within = tv(user_equal_weight_dist(ua), user_equal_weight_dist(ub))
        # noise floor: randomly split each shared user's year-a outfits into two halves
        r = rs.rand(len(ua)) < 0.5
        h1, h2 = ua[r], ua[~r]
        both2 = np.intersect1d(h1.user.unique(), h2.user.unique())
        floor = tv(user_equal_weight_dist(h1[h1.user.isin(both2)]), user_equal_weight_dist(h2[h2.user.isin(both2)])) if len(both2) > 20 else float("nan")
        print(f"{a}->{b} | {len(both):,} | {overall:.3f} | {within:.3f} | {floor:.3f}")

run(">=4 distinct categories only (no like filter)", df[df.ncat >= 4])
run("official eligible (>=4 categories AND like_num>=50)", df[(df.ncat >= 4) & (df.like >= 50)])
