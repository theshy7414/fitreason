"""NEW script (final report). Read-only: loads saved outputs (scores.npz, main_bootstrap.pkl, instance table) and writes ONLY into final_report_assets/.
Run: PYTHONPATH=/home/chi/outfitmatch-research/step0/pylibs python3 make_assets.py
No model is trained; no existing file is modified."""
import os, pickle, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/chi/outfitmatch-research/step0"
OUT = os.path.dirname(os.path.abspath(__file__))
SETS = ["test_id", "test_ud", "test_2014", "test_2015", "test_2017"]
LAB = ["test_id", "test_ud", "2014", "2015", "2017"]
B = pickle.load(open(f"{ROOT}/runs/exp2/main_bootstrap.pkl", "rb"))   # (fam, set) -> (res, base, bb, n, nusers); res[name] = (point dict, bootstrap dict)
ci = lambda a: tuple(np.percentile(np.asarray(a)[~np.isnan(a)], [2.5, 97.5]))

# ---------- Fig 1: AUROC by method, S and E family, with user-cluster 95% CI
meth = {"S": ["msp_single", "ens_msp", "ens_entropy", "ens_vote_peers", "ens_mi", "knn_pair", "meta"],
        "E": ["ens_msp", "ens_entropy", "ens_vote", "msp_single", "ens_mi", "knn_pair", "meta"]}
cols = plt.cm.tab10.colors
fig, axs = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
for ax, fam, ttl in zip(axs, "SE", ["S-family (error = seed-0 single model)", "E-family (error = 5-seed ensemble)"]):
    for j, m in enumerate(meth[fam]):
        pts, lo, hi = [], [], []
        for s in SETS:
            pt, bs = B[(fam, s)][0][m]; l, h = ci(bs["auc"]); pts.append(pt["auc"]); lo.append(pt["auc"] - l); hi.append(h - pt["auc"])
        x = np.arange(len(SETS)) + (j - 3) * 0.1
        ax.errorbar(x, pts, yerr=[lo, hi], fmt="o", ms=4, capsize=2, color=cols[j], label=m)
    ax.axhline(0.5, color="k", ls=":", lw=1); ax.set_xticks(range(len(SETS))); ax.set_xticklabels(LAB); ax.set_title(ttl, fontsize=10); ax.grid(alpha=.3)
axs[0].set_ylabel("error-detection AUROC (positive = error)"); axs[0].legend(fontsize=7, ncol=2)
fig.suptitle("Fig 1. Error-detection AUROC by method (user-cluster bootstrap 95% CI, B=2000). Dotted = chance 0.5", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_auroc_by_method.png", dpi=130); plt.close(fig)

# ---------- Fig 2: paired delta AUROC vs family reference, Gate-2 threshold +0.02
fig, axs = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True); rows = []
for ax, fam, ref in zip(axs, "SE", ["msp_single", "ens_msp"]):
    ms = [m for m in meth[fam] if m != ref]
    for j, m in enumerate(ms):
        d, lo, hi = [], [], []
        for s in SETS:
            res = B[(fam, s)][0]; arr = res[m][1]["auc"] - res[ref][1]["auc"]; p = res[m][0]["auc"] - res[ref][0]["auc"]; l, h = ci(arr)
            d.append(p); lo.append(p - l); hi.append(h - p); rows.append((fam, ref, m, s, p, l, h))
        ax.errorbar(np.arange(len(SETS)) + (j - 3) * 0.1, d, yerr=[lo, hi], fmt="o", ms=4, capsize=2, color=cols[j + 1], label=m)
    ax.axhline(0, color="k", lw=1); ax.axhline(0.02, color="r", ls="--", lw=1, label="Gate 2 bar (+0.02)")
    ax.set_xticks(range(len(SETS))); ax.set_xticklabels(LAB); ax.set_title(f"{fam}-family: delta AUROC vs {ref} (paired CI)", fontsize=10); ax.grid(alpha=.3); ax.legend(fontsize=7)
axs[0].set_ylabel("delta AUROC")
fig.suptitle("Fig 2. Paired AUROC difference vs the family reference (MSP). No candidate reaches the +0.02 Gate-2 bar.", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_paired_delta_auroc.png", dpi=130); plt.close(fig)
with open(f"{OUT}/paired_delta_auroc.csv", "w") as f:
    f.write("family,reference,method,set,delta_auroc,ci_lo,ci_hi\n")
    for r in rows: f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.4f},{r[5]:.4f},{r[6]:.4f}\n")

# ---------- Fig 3: accuracy / calibration / AUROC across sets (5-seed mean), recomputed from saved logits
sc = np.load(f"{ROOT}/runs/exp2/scores.npz"); tab = np.load(f"{ROOT}/results/exp1_instance_table_taskB.npz", allow_pickle=True)
sm = lambda l: (lambda e: e / e.sum(1, keepdims=True))(np.exp(l - l.max(1, keepdims=True)))
def ece(p, y, bins=15):
    c = p.max(1); ok = (p.argmax(1) == y).astype(float); o = np.argsort(c, kind="mergesort"); e = 0
    for ch in np.array_split(o, bins): e += len(ch) / len(y) * abs(ok[ch].mean() - c[ch].mean())
    return e
acc, eces, cmA, auroc_pts = [], [], [], []
for r in SETS:
    y = tab[f"{r}/label"]; ps = [sm(sc[f"s{s}/{r}/logits"]) for s in range(5)]
    acc.append(np.mean([(p.argmax(1) == y).mean() for p in ps]) * 100); eces.append(np.mean([ece(p, y) for p in ps]))
    cmA.append(np.mean([(p.max(1).mean() - (p.argmax(1) == y).mean()) for p in ps]) * 100)
lin = [53.6, 51.5, 43.0, 40.7, 39.8]; cosv = [34.3, 32.4, 29.3, 28.6, 28.5]       # from results/exp2_linear_baseline_report.txt (5-seed means)
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].plot(LAB, acc, "o-", label="Visual MLP (5-seed mean)"); axs[0].plot(LAB, lin, "s--", label="linear-encoder bilinear (final grid)"); axs[0].plot(LAB, cosv, "^:", label="cosine (training-free)")
axs[0].axhline(25, color="k", ls=":", lw=1); axs[0].set_title("Accuracy % (chance 25)", fontsize=10); axs[0].legend(fontsize=7); axs[0].grid(alpha=.3)
axs[1].bar(LAB, eces, color="C1"); axs[1].set_title("ECE (15 equal-mass bins, mean of 5 seeds)", fontsize=10)
for i, v in enumerate(cmA): axs[1].text(i, eces[i] + 0.002, f"conf-acc {v:+.1f}pp", ha="center", fontsize=7)
a_pts = [B[("S", s)][0]["msp_single"][0]["auc"] for s in SETS]; a_ci = [ci(B[("S", s)][0]["msp_single"][1]["auc"]) for s in SETS]
axs[2].errorbar(range(5), a_pts, yerr=[[p - c[0] for p, c in zip(a_pts, a_ci)], [c[1] - p for p, c in zip(a_pts, a_ci)]], fmt="o-", capsize=3)
axs[2].set_xticks(range(5)); axs[2].set_xticklabels(LAB); axs[2].set_title("MSP error-detection AUROC (seed 0, 95% CI)", fontsize=10); axs[2].axhline(0.5, color="k", ls=":", lw=1); axs[2].grid(alpha=.3)
fig.suptitle("Fig 3. Cross-year degradation: accuracy, calibration, error detection (descriptive)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_cross_year_summary.png", dpi=130); plt.close(fig)

# ---------- Fig 4: fixed val_id threshold, S-family and E-family kept SEPARATE (errors of different predictors)
fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for i, (fam, ms) in enumerate([("S", ["msp_single", "knn_pair", "ens_mi"]), ("E", ["ens_msp", "ens_mi", "knn_pair"])]):
    for m in ms:
        cov = [B[(fam, s)][0][m][0]["cov"] * 100 for s in SETS]; rk = [B[(fam, s)][0][m][0]["risk_fix"] * 100 for s in SETS]
        axs[i, 0].plot(LAB, cov, "o-", label=m); axs[i, 1].plot(LAB, rk, "o-", label=m)
    axs[i, 0].set_title(f"{fam}-family: coverage % at val_id tau (target 80)", fontsize=9); axs[i, 1].set_title(f"{fam}-family: risk % among accepted", fontsize=9)
    axs[i, 0].legend(fontsize=7); axs[i, 0].grid(alpha=.3); axs[i, 1].grid(alpha=.3)
fig.suptitle("Fig 4. Fixed val_id threshold transfer (point estimates), S- and E-family in separate rows", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_fixed_threshold_by_family.png", dpi=130); plt.close(fig)

# ---------- CSV: seed-mean set-level metrics + counts
with open(f"{OUT}/set_level_summary.csv", "w") as f:
    f.write("set,n_instances,n_users,effective_users_kish,acc_5seed_mean_pct,ece_5seed_mean_arraysplit,conf_minus_acc_pp\n")
    for r, a, e, c in zip(SETS, acc, eces, cmA):
        u = tab[f"{r}/user_id"]; _, cnt = np.unique(u, return_counts=True)
        f.write(f"{r},{len(u)},{len(cnt)},{cnt.sum()**2/(cnt**2).sum():.0f},{a:.2f},{e:.4f},{c:.2f}\n")
print("done")
