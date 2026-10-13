"""Phase 2 report (no training): accuracy, baselines, paired comparisons, shift decomposition, audit-flag subgroups, NLL/ECE preview.
Run: PYTHONPATH=./pylibs:./scripts python3 scripts/53_phase2_report.py | tee results/exp1_phase2_report.txt
"""
import json
import numpy as np
from exp1_common import *

PH2 = os.path.join(ROOT, "runs", "phase2")
tab = np.load(os.path.join(RES, "exp1_instance_table_taskB.npz"))
roles = ["test_id", "test_ud", "test_2014", "test_2015", "test_2017"]
variants = ["visual", "visual_cat"]
T = lambda r, k: tab[f"{r}/{k}"]
label = {r: T(r, "label") for r in roles + ["val_id"]}
users = {r: T(r, "user_id") for r in roles + ["val_id"]}
logits = {v: {s: np.load(os.path.join(PH2, f"{v}_seed{s}", "preds.npz")) for s in range(5)} for v in variants}
corr = {v: {s: {r: (logits[v][s][r].argmax(1) == label[r]).astype(float) for r in roles + ["val_id"]} for s in range(5)} for v in variants}
avg = {v: {r: np.mean([corr[v][s][r] for s in range(5)], 0) for r in roles + ["val_id"]} for v in variants}
base = {b: {r: T(r, f"acc_{b}") for r in roles} for b in ["cosine", "price", "cat2prior", "popularity", "cooc"]}
B = 2000


def stats(v, u):
    uu, inv = np.unique(u, return_inverse=True); return uu, np.bincount(inv, weights=v), np.bincount(inv).astype(float)


def diff_ci(va, ua, vb, ub, seed=0, paired_same=False):
    """mean(va) - mean(vb); independent user-cluster bootstrap of the two sets (or one set if paired_same)."""
    rs = np.random.RandomState(seed)
    if paired_same:
        x = va - vb; return cluster_boot(x, ua, B=B, seed=seed)
    _, sa, na = stats(va, ua); _, sb, nb = stats(vb, ub); d = np.empty(B)
    for k in range(B):
        i = rs.randint(len(sa), size=len(sa)); j = rs.randint(len(sb), size=len(sb))
        d[k] = sa[i].sum() / na[i].sum() - sb[j].sum() / nb[j].sum()
    return va.mean() - vb.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def sub_ci(v, u, m):
    if m.sum() < 50 or len(set(u[m])) < 20:
        return None
    return cluster_boot(v[m], u[m], B=B)


def f(x):
    return "          -          " if x is None else fmt(x)


def ece(p, y, bins=15):
    conf = p.max(1); ok = (p.argmax(1) == y).astype(float); o = np.argsort(conf); n = len(y); e = 0.0
    for ch in np.array_split(o, bins):
        e += len(ch) / n * abs(ok[ch].mean() - conf[ch].mean())
    return e


def sm(l):
    e = np.exp(l - l.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


# ----------------------------------------------------------------------------------------------------------
H("TABLE 1. Accuracy % (chance 25). Models: seed-averaged per-instance correctness, user-cluster 95% CI; [seed min-max] over the 5 seeds")
print(f"{'':<14}" + "".join(f"{r:>34}" for r in roles))
for v in variants:
    row = f"{'MLP '+v:<14}"
    for r in roles:
        per = [corr[v][s][r].mean() * 100 for s in range(5)]
        row += f"{fmt(cluster_boot(avg[v][r], users[r], B=B)):>22} [{min(per):4.1f}-{max(per):4.1f}]  "
    print(row)
for b in base:
    print(f"{b:<14}" + "".join(f"{fmt(cluster_boot(base[b][r], users[r], B=B)):>34}" for r in roles))
print("n instances / users / effective users: " + " | ".join(f"{r}: {len(label[r])}/{len(set(users[r]))}/{eff_users(users[r])[0]:.0f}" for r in roles))

H("TABLE 2. Paired differences in accuracy (pp) [user-cluster 95% CI], seed-averaged model minus comparator")
print(f"{'comparison':<28}" + "".join(f"{r:>26}" for r in roles))
for v in variants:
    for b in base:
        print(f"{v+' - '+b:<28}" + "".join(f"{fmt(diff_ci(avg[v][r], users[r], base[b][r], users[r], paired_same=True)):>26}" for r in roles))
print(f"{'visual_cat - visual':<28}" + "".join(f"{fmt(diff_ci(avg['visual_cat'][r], users[r], avg['visual'][r], users[r], paired_same=True)):>26}" for r in roles))

# ----------------------------------------------------------------------------------------------------------
H("TABLE 3. DECOMPOSITION of the drop from a same-year reference to each shifted year (pp, [95% CI]; independent cluster bootstraps)")
print("  accuracy degradation        = model acc(shift) - model acc(reference)")
print("  benchmark difficulty shift  = same difference for TRAINING-FREE cosine / price (nothing learned, so not model failure)")
print("  memorisation channel        = same difference for the train co-occurrence baseline (id-level pair memory)")
print("  excess degradation          = drop of (model - cosine): what is left after subtracting the training-free drift")
for ref in ("test_id", "test_ud"):
    print(f"\nreference = {ref}")
    print(f"{'shift year':<11}{'model(visual)':>22}{'model(vis+cat)':>22}{'cosine':>22}{'price':>22}{'cooc':>20}{'excess(visual)':>22}{'excess(vis+cat)':>22}")
    for y in ("test_2014", "test_2015", "test_2017"):
        row = f"{y:<11}"
        for nm, ay, ar in [("v", avg["visual"][y], avg["visual"][ref]), ("c", avg["visual_cat"][y], avg["visual_cat"][ref]),
                           ("cos", base["cosine"][y], base["cosine"][ref]), ("pr", base["price"][y], base["price"][ref]), ("co", base["cooc"][y], base["cooc"][ref])]:
            row += f"{fmt(diff_ci(ay, users[y], ar, users[ref])):>22}"
        for vv in variants:
            row += f"{fmt(diff_ci(avg[vv][y] - base['cosine'][y], users[y], avg[vv][ref] - base['cosine'][ref], users[ref])):>22}"
        print(row)

# ----------------------------------------------------------------------------------------------------------
H("TABLE 4. SUBGROUPS by audit flags (seed-averaged MLP acc % [CI]; '-' = fewer than 50 instances or 20 users). cos/cooc = training-free references")
strata = {0: "id-seen positive", 1: "twin-only (id unseen, visual twin)", 2: "visually novel positive"}
for r in roles:
    st = T(r, "stratum"); dec = T(r, "decidable"); lk = T(r, "id_leak50"); nd = T(r, "nd_flagged"); us = T(r, "user_seen")
    clean = ~dec & ~lk.astype(bool) & ~nd.astype(bool)
    groups = [("ALL", np.ones(len(st), bool))] + [(strata[k], st == k) for k in range(3)] + [
        ("co-occurrence decidable", dec.astype(bool)), ("not decidable", ~dec.astype(bool)), ("partial-overlap outfit (>=50% ids)", lk.astype(bool)),
        ("near-dup flagged", nd.astype(bool)), ("CLEAN (not decidable/overlap/flagged)", clean),
        ("user seen in train", us.astype(bool)), ("user unseen in train", ~us.astype(bool)),
        ("novel positive AND clean", (st == 2) & clean)]
    print(f"\n[{r}]\n{'group':<40}{'n':>6}{'users':>7}{'MLP visual':>22}{'MLP vis+cat':>22}{'cosine':>22}{'cooc':>22}")
    for nm, m in groups:
        if m.sum() < 50:
            print(f"{nm:<40}{int(m.sum()):>6}"); continue
        print(f"{nm:<40}{int(m.sum()):>6}{len(set(users[r][m])):>7}{f(sub_ci(avg['visual'][r], users[r], m)):>22}{f(sub_ci(avg['visual_cat'][r], users[r], m)):>22}"
              f"{f(sub_ci(base['cosine'][r], users[r], m)):>22}{f(sub_ci(base['cooc'][r], users[r], m)):>22}")

H("TABLE 5. MATCHED COMPARISON: only instances whose positive is visually novel AND clean (no co-occurrence signal, no overlap, no near-dup flag)")
print("Removes the 'seen item / memorised pair' advantage the same-year sets have; remaining drop = shift in visual content / task difficulty / model.")
print(f"{'':<12}{'n':>6}{'users':>7}{'MLP visual':>22}{'MLP vis+cat':>22}{'cosine':>22}{'price':>22}{'cooc':>20}")
M5 = {}
for r in roles:
    m = (T(r, "stratum") == 2) & ~T(r, "decidable").astype(bool) & ~T(r, "id_leak50").astype(bool) & ~T(r, "nd_flagged").astype(bool); M5[r] = m
    print(f"{r:<12}{int(m.sum()):>6}{len(set(users[r][m])):>7}{f(sub_ci(avg['visual'][r], users[r], m)):>22}{f(sub_ci(avg['visual_cat'][r], users[r], m)):>22}"
          f"{f(sub_ci(base['cosine'][r], users[r], m)):>22}{f(sub_ci(base['price'][r], users[r], m)):>22}{f(sub_ci(base['cooc'][r], users[r], m)):>20}")
print("\nmatched drop vs test_id matched (pp [CI]), visual MLP / cosine:")
for y in ("test_ud", "test_2014", "test_2015", "test_2017"):
    a = diff_ci(avg["visual"][y][M5[y]], users[y][M5[y]], avg["visual"]["test_id"][M5["test_id"]], users["test_id"][M5["test_id"]])
    c = diff_ci(base["cosine"][y][M5[y]], users[y][M5[y]], base["cosine"]["test_id"][M5["test_id"]], users["test_id"][M5["test_id"]])
    print(f"  {y:<10} MLP {fmt(a)}   cosine {fmt(c)}")

# ----------------------------------------------------------------------------------------------------------
H("TABLE 6. PREVIEW ONLY (reliability analysis comes after discussion): NLL / ECE(15 equal-mass bins) / mean max-prob, mean over 5 seeds")
print(f"{'':<14}{'metric':<8}" + "".join(f"{r:>12}" for r in roles))
for v in variants:
    for mname in ("acc%", "NLL", "ECE", "conf%"):
        vals = []
        for r in roles:
            xs = []
            for s in range(5):
                p = sm(logits[v][s][r]); y = label[r]
                xs.append({"acc%": (p.argmax(1) == y).mean() * 100, "NLL": -np.log(p[np.arange(len(y)), y] + 1e-12).mean(), "ECE": ece(p, y), "conf%": p.max(1).mean() * 100}[mname])
            vals.append(np.mean(xs))
        print(f"{'MLP '+v:<14}{mname:<8}" + "".join(f"{x:>12.3f}" if mname in ("NLL", "ECE") else f"{x:>12.1f}" for x in vals))

# ----------------------------------------------------------------------------------------------------------
H("FIGURES")
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(os.path.join(PH2, "figures"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4)); w = 0.14; xs = np.arange(len(roles))
    series = [("MLP visual", {r: avg["visual"][r] for r in roles}), ("MLP vis+cat", {r: avg["visual_cat"][r] for r in roles})] + \
             [(b, base[b]) for b in ("cosine", "price", "cooc")]
    for k, (nm, d) in enumerate(series):
        ci = [cluster_boot(d[r], users[r], B=300) for r in roles]
        ax.bar(xs + k * w, [c[0] * 100 for c in ci], w, yerr=[[ (c[0] - c[1]) * 100 for c in ci], [(c[2] - c[0]) * 100 for c in ci]], label=nm, capsize=2)
    ax.axhline(25, ls=":", c="gray"); ax.set_xticks(xs + 2 * w); ax.set_xticklabels(roles); ax.set_ylabel("accuracy %"); ax.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(PH2, "figures", "accuracy_by_set.png"), dpi=130)
    log = [json.loads(l) for l in open(os.path.join(ROOT, "runs", "phase1", "grid3", "log.jsonl"))]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6)); e = [r["epoch"] for r in log]
    ax[0].plot(e, [r["train_loss"] for r in log], label="train loss (online, dropout on)"); ax[0].plot(e, [r["probe_nll"] for r in log], label="train-probe NLL")
    ax[0].plot(e, [r["val_nll"] for r in log], label="val_id NLL"); ax[0].axhline(np.log(4), ls=":", c="gray"); ax[0].legend(fontsize=7); ax[0].set_xlabel("epoch")
    ax[1].plot(e, [r["probe_acc"] * 100 for r in log], label="train-probe acc"); ax[1].plot(e, [r["val_acc"] * 100 for r in log], label="val_id acc")
    ax[1].axhline(33.0, ls="--", c="r", label="cosine val_id"); ax[1].axhline(25, ls=":", c="gray"); ax[1].legend(fontsize=7); ax[1].set_xlabel("epoch")
    plt.tight_layout(); plt.savefig(os.path.join(PH2, "figures", "phase1_learning_curve.png"), dpi=130)
    print("saved runs/phase2/figures/accuracy_by_set.png, phase1_learning_curve.png")
except Exception as ex:
    print("figures skipped:", ex)
