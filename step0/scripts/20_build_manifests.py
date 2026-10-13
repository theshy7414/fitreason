"""Experiment 1: freeze splits + evaluation instances + required item_id set from METADATA ONLY (no image features).

Outputs (manifests/, all deterministic given the constants below):
  S1_<tag>_*.json.gz          project split (user-held-out ID test, shifted-year tests) + Task B fixed instances
  S0_official_*.json.gz       official-procedure split + Task A set-matching instances (n_cands 4 and 8)
  required_items_v1.txt       union of all item_ids that any frozen set touches
  MANIFEST_SHA256.txt         sha256 of every manifest file
  (results/exp1_manifest_report.txt is produced by 21_audit_and_baselines.py)

Run: PYTHONPATH=./pylibs python3 scripts/20_build_manifests.py
"""
import os, sys, json, gzip, hashlib, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAN = os.path.join(ROOT, "manifests")
os.makedirs(MAN, exist_ok=True)

# ---------------- frozen constants (changing any of them = new manifest version) ----------------
VERSION = "v1"
PROJECT_SEED = 20261012
OFFICIAL_SEED = 0
N_TRAIN = 30_816          # official train size
N_EVAL_ID = 3_851         # official val/test size, used for ID val / ID test / UD test cap
N_EVAL_SHIFT = 7_702      # = official val+test of the test year, used as ONE shifted test set
HELDOUT_MOD = 10          # 1/10 of users (by md5) never appear in project training
N_NEG = 3                 # Task B: 3 negatives -> 4 candidates

t0 = time.time()
Z = np.load(os.path.join(ROOT, "data", "outfits", "outfits_compact.npz"))
set_id, user_id, like, n_items, year, ymd = (Z[k] for k in ["set_id", "user_id", "like", "n_items", "year", "ymd"])
item_id, cat1, cat2, price = (Z[k] for k in ["item_id", "cat1", "cat2", "price"])
N = len(set_id)
off = np.zeros(N + 1, dtype=np.int64); np.cumsum(n_items, out=off[1:])


def items_of(o):
    return item_id[off[o]:off[o + 1]]


# ---------------- eligibility (official filter) ----------------
bit = (1 << (cat1 - cat1.min()).astype(np.int64))
mask_o = np.bitwise_or.reduceat(bit, off[:-1])
um = np.unique(mask_o)
lut = dict(zip(um.tolist(), [bin(int(m)).count("1") for m in um]))
ncat = np.fromiter((lut[int(m)] for m in mask_o), dtype=np.int16, count=N)
elig = (ncat >= 4) & (like >= 50)

# ---------------- dedupe (project split only; official procedure does NOT dedupe) ----------------
el_idx = np.where(elig)[0]
sig = {}
rep_items = np.zeros(N, dtype=bool)
for o in el_idx:
    its = items_of(o)
    if len(np.unique(its)) != len(its):
        rep_items[o] = True
    sig[o] = hashlib.md5(np.sort(its).astype(np.int64).tobytes()).hexdigest()[:16]
df = pd.DataFrame({"o": el_idx, "sig": [sig[o] for o in el_idx], "ymd": ymd[el_idx], "sid": set_id[el_idx]})
df = df.sort_values(["ymd", "sid"], kind="mergesort")
first = ~df.duplicated("sig", keep="first")
n_dup_dropped = int((~first).sum())
keep = np.zeros(N, dtype=bool); keep[df.o.values[first.values]] = True
n_rep_dropped = int((keep & rep_items).sum()); keep &= ~rep_items
pool = {y: np.where(keep & (year == y))[0] for y in range(2010, 2021)}
print(f"eligible={len(el_idx):,} exact-duplicate item-sets dropped={n_dup_dropped:,} outfits with repeated item_id dropped={n_rep_dropped:,}")
print("project pools:", {y: len(pool[y]) for y in range(2013, 2018)})


def held_user(u):
    return int(hashlib.md5(f"outfitmatch-heldout-{VERSION}:{int(u)}".encode()).hexdigest()[:8], 16) % HELDOUT_MOD == 0


def sub_seed(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


# ---------------- Task B instance construction ----------------
def build_task_b(O, role, train_users, train_items, train_freq):
    """One fixed 4-way item-FITB instance per outfit in O. Negatives come ONLY from the same set O (same year, same role),
    same category_id1 as the positive, other users, not in the query outfit, slot-weighted (same marginal as the positive)."""
    O = np.asarray(O)
    slots = np.concatenate([np.arange(off[o], off[o + 1]) for o in O])
    owner = np.repeat(O, n_items[O])
    by_cat = {}
    for c in np.unique(cat1[slots]):
        by_cat[int(c)] = np.where(cat1[slots] == c)[0]
    rows, dropped = [], 0
    for o in sorted(O.tolist(), key=lambda x: int(set_id[x])):
        rs = np.random.RandomState(sub_seed("taskB", VERSION, role, set_id[o], PROJECT_SEED))
        sl = np.arange(off[o], off[o + 1])
        k = int(rs.randint(len(sl)))
        ps = sl[k]; pos = int(item_id[ps]); pc = int(cat1[ps])
        ctx = [int(item_id[s]) for i, s in enumerate(sl) if i != k]
        forbid = set(item_id[sl].tolist())
        pool_idx = by_cat[pc]
        negs, negslots, tries = [], [], 0
        while len(negs) < N_NEG and tries < 5000:
            tries += 1
            j = pool_idx[rs.randint(len(pool_idx))]
            ow = owner[j]; it = int(item_id[slots[j]])
            if ow == o or user_id[ow] == user_id[o] or it in forbid or it in negs:
                continue
            negs.append(it); negslots.append(int(slots[j]))
        if len(negs) < N_NEG:
            dropped += 1
            continue
        cands = [pos] + negs
        cslots = [int(ps)] + negslots
        perm = rs.permutation(4)
        cands = [cands[i] for i in perm]; cslots = [cslots[i] for i in perm]
        label = int(np.where(perm == 0)[0][0])
        iid = hashlib.md5(f"{role}|{set_id[o]}|{cands}".encode()).hexdigest()[:12]
        rows.append(dict(
            instance_id=iid, role=role, year=int(year[o]), set_id=int(set_id[o]), user_id=int(user_id[o]),
            ctx=ctx, cands=cands, label=label, cat1=pc,
            cand_cat2=[int(cat2[s]) for s in cslots], cand_price=[int(price[s]) for s in cslots],
            ctx_cat1=[int(cat1[off[o] + i]) for i in range(len(sl)) if i != k],
            user_seen=bool(user_id[o] in train_users), pos_seen=bool(pos in train_items),
            neg_seen=[bool(i in train_items) for i in cands if i != pos],
            cand_train_freq=[int(train_freq.get(i, 0)) for i in cands]))
    return rows, dropped


def save_gz(name, obj):
    p = os.path.join(MAN, name)
    with gzip.GzipFile(p, "wb", mtime=0) as f:
        for line in obj:
            f.write((json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n").encode())
    return p


def save_json(name, obj):
    p = os.path.join(MAN, name)
    with open(p, "w") as f:
        json.dump(obj, f, sort_keys=True, indent=1)
    return p


def outfit_rows(O, role):
    return [dict(role=role, o=int(o), set_id=int(set_id[o]), user_id=int(user_id[o]), year=int(year[o]), n_items=int(n_items[o]))
            for o in sorted(O.tolist(), key=lambda x: int(set_id[x]))]


required = {}   # group -> set of item ids
sizes = {}


def add_required(group, O):
    s = set()
    for o in O:
        s.update(items_of(o).tolist())
    required[group] = s


# ---------------- S1: project split ----------------
def build_s1(train_year, test_years, tag):
    P = pool[train_year]
    held = np.array([held_user(u) for u in user_id[P]])
    rs = np.random.RandomState(PROJECT_SEED + train_year)
    rest = rs.permutation(P[~held])
    train, tail = rest[:N_TRAIN], rest[N_TRAIN:]
    val, test_id = tail[:N_EVAL_ID], tail[N_EVAL_ID:2 * N_EVAL_ID]
    ud = rs.permutation(P[held])[:N_EVAL_ID]
    sets = {"train": train, "val_id": val, "test_id": test_id, "test_ud": ud}
    for y in test_years:
        rsy = np.random.RandomState(sub_seed("shift-test-sample", VERSION, y, PROJECT_SEED))
        sets[f"test_{y}"] = rsy.permutation(pool[y])[:N_EVAL_SHIFT]
    train_users = set(user_id[train].tolist())
    train_items = set(); freq = {}
    for o in train:
        for i in items_of(o).tolist():
            train_items.add(i); freq[i] = freq.get(i, 0) + 1
    outfits, inst_rows, drops = [], {}, {}
    for role, O in sets.items():
        outfits += outfit_rows(O, role)
        if role != "train":
            inst_rows[role], drops[role] = build_task_b(O, role, train_users, train_items, freq)
            save_gz(f"S1_{tag}_taskB_{role}.jsonl.gz", inst_rows[role])
        add_required(f"S1_{tag}:{role}", O)
    save_gz(f"S1_{tag}_outfits.jsonl.gz", outfits)
    sizes[tag] = {r: len(O) for r, O in sets.items()}
    sizes[tag + "_taskB_n"] = {r: len(v) for r, v in inst_rows.items()}
    sizes[tag + "_taskB_dropped"] = drops
    print(tag, "outfits:", sizes[tag], "| TaskB instances:", sizes[tag + "_taskB_n"], "| dropped:", drops)
    return sets


s1_main = build_s1(2013, [2014, 2015, 2017], "tr2013")
s1_rep = build_s1(2014, [2015, 2016, 2017], "tr2014")   # replication axis; evaluated only if Gate 1 passes


# ---------------- S0: official procedure (Task A) ----------------
def official_sets(ys, ye_list, seed=OFFICIAL_SEED):
    idx = np.where(elig)[0]                      # file order, as in the official DataFrame
    out = {}
    ys_idx = idx[year[idx] == ys]
    perm = pd.Series(ys_idx).sample(frac=1, random_state=seed).values
    train = perm[:N_TRAIN]
    for ye in ye_list:
        if ye == ys:
            tail = perm[N_TRAIN:]
            val, test = train_test_split(tail, test_size=0.5, random_state=seed)
        else:
            ye_idx = idx[year[idx] == ye]
            shuffled = pd.Series(ye_idx).sample(frac=1, random_state=seed).values[:2 * N_EVAL_ID]
            val, test = train_test_split(shuffled, test_size=0.5, random_state=seed)
        out[ye] = (val, test)
    return train, out


def official_test_examples(test_sets, n_cands, seed=OFFICIAL_SEED, n_comb=1):
    """Same algorithm / RNG call order as shift15m IQONOutfits._make_test_examples (n_comb=1)."""
    np.random.seed(seed)
    ex = []
    n = len(test_sets)
    for i in range(n):
        lst = np.delete(np.arange(n), i)
        others = np.random.choice(lst, n_comb - 1, replace=False).tolist()
        target = [i] + others
        setX, setY = [], []
        for j in target:
            its = np.array(items_of(test_sets[j]))
            y_size = len(its) // 2
            xy_mask = np.random.permutation([True] * (len(its) - y_size) + [False] * y_size)
            setX.extend(its[xy_mask].tolist()); setY.extend(its[~xy_mask].tolist())
        answers = [setY]
        for _ in range(n_cands - 1):
            lst = np.delete(np.arange(n), target)
            negatives = np.random.choice(lst, n_comb, replace=False).tolist()
            target += negatives
            sy = []
            for k in negatives:
                its = np.random.permutation(items_of(test_sets[k]))
                y_size = len(its) // 2
                sy.extend(its[:y_size].tolist())
            answers.append(sy)
        ex.append((setX, answers))
    return ex


def build_s0(ys, ye_list, tag):
    train, evals = official_sets(ys, ye_list)
    train_users = set(user_id[train].tolist())
    outfits = outfit_rows(train, "train")
    add_required(f"S0:train{ys}", train)
    tsizes = {"train": len(train)}
    for ye, (val, test) in evals.items():
        outfits += outfit_rows(val, f"val_{ye}") + outfit_rows(test, f"test_{ye}")
        add_required(f"S0:test_{ye}", test); add_required(f"S0:val_{ye}", val)
        tsizes[f"val_{ye}"] = len(val); tsizes[f"test_{ye}"] = len(test)
        for nc in (4, 8):
            ex = official_test_examples(test, nc)
            rs = np.random.RandomState(sub_seed("taskA-shuffle", VERSION, ye, nc, PROJECT_SEED))
            rows = []
            for i, (x, answers) in enumerate(ex):
                perm = rs.permutation(nc)
                shuffled = [answers[j] for j in perm]
                label = int(np.where(perm == 0)[0][0])
                o = int(test[i])
                rows.append(dict(role=f"test_{ye}", n_cands=nc, year=int(year[o]), set_id=int(set_id[o]), user_id=int(user_id[o]),
                                 query=x, answers=shuffled, label=label, official_label0=True,
                                 user_seen=bool(user_id[o] in train_users)))
            save_gz(f"S0_{tag}_taskA_nc{nc}_test{ye}.jsonl.gz", rows)
    save_gz(f"S0_{tag}_outfits.jsonl.gz", outfits)
    sizes[tag] = tsizes
    print(tag, tsizes)
    return train, evals


s0_train, s0_evals = build_s0(2013, [2013, 2014, 2015, 2017], "official_tr2013")

# ---------------- required item set ----------------
all_items = sorted(set().union(*required.values()))
with open(os.path.join(MAN, f"required_items_{VERSION}.txt"), "w") as f:
    f.write("\n".join(map(str, all_items)) + "\n")
grp = {g: len(s) for g, s in sorted(required.items())}
core = set().union(*[s for g, s in required.items() if g.startswith("S1_tr2013")])
core_o = set().union(*[s for g, s in required.items() if g.startswith("S0")])
rep = set().union(*[s for g, s in required.items() if g.startswith("S1_tr2014")])
summary = dict(version=VERSION, project_seed=PROJECT_SEED, official_seed=OFFICIAL_SEED, sizes=sizes, required_items_by_group=grp,
               required_items_total=len(all_items), required_S1_tr2013=len(core), required_S0=len(core_o), required_S1_tr2014=len(rep),
               dedupe=dict(exact_dup_dropped=n_dup_dropped, repeated_item_outfits_dropped=n_rep_dropped),
               fp16_GB=len(all_items) * 4096 * 2 / 1e9, fp32_GB=len(all_items) * 4096 * 4 / 1e9)
save_json("manifest_summary.json", summary)
with open(os.path.join(MAN, "MANIFEST_SHA256.txt"), "w") as f:
    for fn in sorted(os.listdir(MAN)):
        if fn == "MANIFEST_SHA256.txt":
            continue
        f.write(hashlib.sha256(open(os.path.join(MAN, fn), "rb").read()).hexdigest() + "  " + fn + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "sizes"}, indent=1))
print(f"done in {time.time()-t0:.0f}s")
