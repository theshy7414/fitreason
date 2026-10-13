"""Shared helpers for Experiment 1 pre-training checks. Reads frozen manifests; never re-samples anything."""
import os, json, gzip, collections
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAN = os.path.join(ROOT, "manifests")
FEAT = os.path.join(ROOT, "data", "features_v1")
RES = os.path.join(ROOT, "results")

Z = np.load(os.path.join(ROOT, "data", "outfits", "outfits_compact.npz"))
set_id, user_id, n_items, year = (Z[k] for k in ["set_id", "user_id", "n_items", "year"])
item_id, cat1, cat2, price = (Z[k] for k in ["item_id", "cat1", "cat2", "price"])
off = np.zeros(len(set_id) + 1, dtype=np.int64); np.cumsum(n_items, out=off[1:])
_u, _f = np.unique(item_id, return_index=True)
item_cat1 = dict(zip(_u.tolist(), cat1[_f].tolist()))
item_cat2 = dict(zip(_u.tolist(), cat2[_f].tolist()))
item_price = dict(zip(_u.tolist(), price[_f].tolist()))


def items_of(o):
    return item_id[off[o]:off[o + 1]]


def rd(name):
    with gzip.open(os.path.join(MAN, name), "rt") as f:
        return [json.loads(l) for l in f]


def roles_from_outfits(name):
    d = collections.defaultdict(list)
    for r in rd(name):
        d[r["role"]].append(r["o"])
    return {k: np.array(v) for k, v in d.items()}


def items_in(outfits):
    return np.unique(np.concatenate([items_of(o) for o in outfits]))


def load_required_features():
    """Required items only (391k x 4096 fp16 = 3.2 GB) cached next to the full matrix."""
    cp, ci = os.path.join(FEAT, "required_fp16.npy"), os.path.join(FEAT, "required_ids.npy")
    if os.path.exists(cp):
        return np.load(cp), np.load(ci)
    ids = np.load(os.path.join(FEAT, "item_ids.npy"))
    M = np.load(os.path.join(FEAT, "features_fp16.npy"), mmap_mode="r")
    req = np.array(sorted(int(x) for x in open(os.path.join(MAN, "required_items_v1.txt")).read().split()), dtype=np.int64)
    rows = np.searchsorted(ids, req); assert (ids[rows] == req).all()
    F = np.empty((len(req), M.shape[1]), dtype=np.float16)
    for s in range(0, len(req), 20000):
        F[s:s + 20000] = M[rows[s:s + 20000]]
    np.save(cp, F); np.save(ci, req)
    return F, req


def cluster_boot(vals, users, B=1000, seed=0):
    u, inv = np.unique(users, return_inverse=True)
    s = np.bincount(inv, weights=vals); n = np.bincount(inv).astype(float)
    rs = np.random.RandomState(seed)
    m = np.empty(B)
    for b in range(B):
        i = rs.randint(len(u), size=len(u)); m[b] = s[i].sum() / n[i].sum()
    return float(np.mean(vals)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fmt(x, scale=100.0):
    m, lo, hi = x
    return f"{m*scale:5.1f} [{lo*scale:5.1f},{hi*scale:5.1f}]"


def tie_acc(scores, label):
    scores = np.asarray(scores, dtype=float)
    best = scores >= scores.max() - 1e-12
    return (1.0 / best.sum()) if best[label] else 0.0


def eff_users(users):
    c = np.array(list(collections.Counter(users).values()), dtype=float)
    return (c.sum() ** 2) / (c ** 2).sum(), len(c)


def H(t):
    print("\n" + "=" * 8, t, "=" * 8, flush=True)
