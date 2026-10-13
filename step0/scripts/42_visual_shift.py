"""Check 2 - visual feature shift controlling category_id1, with user-composition controls. Frozen S1 train-2013 sets only.

Domain classifier per category_id1: items of the 2013 TRAIN outfits (domain 0) vs items of an eval set (domain 1).
 * user-grouped 5-fold CV (a user never in train and test folds), per-user cap of 10 items per domain, PCA-128 + logistic regression
 * setting B (between users): all users;  setting W (within users): only users that have items in BOTH domains
   (a user's style/background cannot separate the domains in W)
 * variant U: domain-1 items whose item_id is not in train;  variant N: also visually novel (top-1 centred cosine to a train item < 0.98)
 * controls: test_id (same year, held-out outfits) and test_ud (same year, held-out users) should give AUROC ~ 0.5
Also: characterise exact visual twins (different item_id, cosine >= 0.999).
Run: PYTHONPATH=./pylibs:./scripts python3 scripts/42_visual_shift.py | tee results/exp1_check2_visual_shift.txt
"""
import warnings; warnings.filterwarnings("ignore")
import collections
import numpy as np, torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from exp1_nn import *

CAP_USER, CAP_CLASS, B_BOOT, NPC = 10, 8000, 300, 128
roles1 = roles_from_outfits("S1_tr2013_outfits.jsonl.gz")
train1 = roles1.pop("train")
eval_roles = ["test_id", "test_ud", "test_2014", "test_2015", "test_2017"]
evo = np.concatenate([roles1[r] for r in eval_roles])
T, U, mu, nn_s, nn_i, ref = nn_tables(train1, evo)
Tset = set(T.tolist()); posU = {int(x): k for k, x in enumerate(U)}
cats = sorted(set(item_cat1[int(i)] for i in T))


def item_user_map(outfits):
    d = {}
    for o in sorted(outfits.tolist(), key=lambda x: int(set_id[x])):
        for it in items_of(o).tolist():
            d.setdefault(int(it), int(user_id[o]))
    return d


tr_user = item_user_map(train1)

# ------------------------------------------------------------ twin characterisation (train -> other train)
print("=" * 8, "exact visual twins (different item_id, centred cosine >= 0.999), sample of 3000 train items per category", "=" * 8)
tot = same2 = samep = n_tw = 0
rs = np.random.RandomState(1)
for c in cats:
    Tc = T[np.array([item_cat1[i] for i in T]) == c]
    sub = rs.choice(len(Tc), min(3000, len(Tc)), replace=False)
    Tt = torch.from_numpy(normed(Tc, mu)).half().to(dev); tid = torch.from_numpy(Tc).to(dev)
    v, i = topk_sims(normed(Tc[sub], mu).astype(np.float16), Tc[sub], Tt, tid, k=1)
    for a, b, s in zip(Tc[sub], i[:, 0], v[:, 0]):
        tot += 1
        if s >= 0.999:
            n_tw += 1; same2 += item_cat2[int(a)] == item_cat2[int(b)]
            pa, pb = item_price[int(a)], item_price[int(b)]; samep += (pa == pb and pa > 0)
print(f"twins: {n_tw}/{tot} ({n_tw/tot*100:.1f}%) of sampled train items | same category_id2: {same2/n_tw*100:.1f}% | same price (>0): {samep/n_tw*100:.1f}%")
print("=> twins are (nearly) the same product image re-listed under another item_id; 'seen' by item_id under-counts visual familiarity.")


def wauc(y, s, w):
    o = np.argsort(s, kind="mergesort"); y = y[o]; w = w[o]
    wp = w * y; wn = w * (1 - y)
    if wp.sum() == 0 or wn.sum() == 0:
        return np.nan
    cn = np.cumsum(wn) - wn
    return float((wp * cn).sum() / (wp.sum() * wn.sum()))


def build(role, c, setting, variant, rs):
    d1_all = items_in(roles1[role])
    d1 = [int(i) for i in d1_all if item_cat1[int(i)] == c and int(i) not in Tset]
    if variant == "N":
        d1 = [i for i in d1 if nn_s[posU[i], 0] < 0.98]
    u1_map = item_user_map(roles1[role])
    d0 = [int(i) for i in T if item_cat1[int(i)] == c]
    if setting == "W":
        common = {tr_user[i] for i in d0} & {u1_map[i] for i in d1}
        d0 = [i for i in d0 if tr_user[i] in common]; d1 = [i for i in d1 if u1_map[i] in common]

    def cap(items, umap):
        by = collections.defaultdict(list)
        for i in items:
            by[umap[i]].append(i)
        out = []
        for u, l in by.items():
            out += [l[j] for j in rs.permutation(len(l))[:CAP_USER]]
        if len(out) > CAP_CLASS:
            out = [out[j] for j in rs.permutation(len(out))[:CAP_CLASS]]
        return out
    d0 = cap(d0, tr_user); d1 = cap(d1, u1_map)
    return d0, d1, tr_user, u1_map


pca = {}
for c in cats:
    Tc = T[np.array([item_cat1[i] for i in T]) == c]
    sub = Tc[np.random.RandomState(0).permutation(len(Tc))[:20000]]
    pca[c] = PCA(NPC, random_state=0).fit(normed(sub, mu))


def run_cell(role, c, setting, variant):
    rs = np.random.RandomState(abs(hash((role, c, setting, variant))) % (2 ** 31) if False else (c * 7919 + len(role) * 31 + (setting == "W") * 5 + (variant == "N") * 3))
    d0, d1, m0, m1 = build(role, c, setting, variant, rs)
    if min(len(d0), len(d1)) < 150:
        return None
    items = np.array(d0 + d1); y = np.r_[np.zeros(len(d0)), np.ones(len(d1))]
    users = np.array([m0[i] for i in d0] + [m1[i] for i in d1])
    X = pca[c].transform(normed(items, mu)); X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, users):
        clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=300).fit(X[tr], y[tr])
        oof[te] = clf.decision_function(X[te])
    return dict(y=y, s=oof, users=users, n0=len(d0), n1=len(d1))


def pooled_ci(cells, B=B_BOOT, seed=0):
    allu = np.unique(np.concatenate([c["users"] for c in cells.values()]))
    uidx = {u: k for k, u in enumerate(allu)}
    ui = {c: np.array([uidx[u] for u in v["users"]]) for c, v in cells.items()}
    point = np.mean([wauc(v["y"], v["s"], np.ones(len(v["y"]))) for v in cells.values()])
    rs = np.random.RandomState(seed); bs = []
    for _ in range(B):
        cnt = np.bincount(rs.randint(len(allu), size=len(allu)), minlength=len(allu)).astype(float)
        a = [wauc(v["y"], v["s"], cnt[ui[c]]) for c, v in cells.items()]
        bs.append(np.nanmean(a))
    return point, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


results = {}
for setting in ("B", "W"):
    for variant in ("U", "N"):
        print("\n" + "=" * 8, f"setting {setting} ({'between users' if setting=='B' else 'WITHIN users present in both domains'}), "
              f"variant {variant} ({'unseen item_id' if variant=='U' else 'unseen id AND centred-cos<0.98 to every train item'})", "=" * 8)
        print(f"{'eval set':<10}{'pooled AUROC (equal-weight over categories) [user-cluster 95% CI]':>66}   per-category AUROC (n_dom1)")
        for role in eval_roles:
            cells = {}
            for c in cats:
                r = run_cell(role, c, setting, variant)
                if r is not None:
                    cells[c] = r
            if len(cells) < 4:
                print(f"{role:<10}{'- (too few items/categories)':>66}"); continue
            p, lo, hi = pooled_ci(cells)
            per = "  ".join(f"{c}:{wauc(v['y'], v['s'], np.ones(len(v['y']))):.2f}({v['n1']})" for c, v in cells.items())
            nus = len(set(np.concatenate([v['users'] for v in cells.values()])))
            print(f"{role:<10}{p:>8.3f} [{lo:.3f},{hi:.3f}]  users={nus:<5}  {per}")
            results[(setting, variant, role)] = (p, lo, hi)
print("\n(0.5 = indistinguishable from the 2013 train domain. test_id/test_ud are same-year controls: they show the floor produced by the procedure itself.)")
