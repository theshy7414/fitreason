"""Supplement (after the reliability experiments, not used by them): simple linear baseline for DESCRIPTIVE baseline-relative degradation.
Model: linear encoder (4096->256, dropout, NO nonlinearity), mean-pooled query, score = <c,e> + w.e  (bilinear/low-rank). Trained on train-2013 only;
hyperparameters (dropout, lr) chosen on val_id NLL only (seed 0); 5 seeds with the chosen config. No shifted-year data used for any choice.
Run: PYTHONPATH=./pylibs:./scripts python3 -u scripts/63_linear_baseline.py 2>&1 | grep -v -i warn | tee results/exp2_linear_baseline_report.txt
"""
import json, itertools
import numpy as np
from taskb_lib import *
from exp2_lib import boot_weights, ci

OUT = os.path.join(ROOT, "runs", "exp2", "linear"); os.makedirs(OUT, exist_ok=True)
roles = ["val_id", "test_id", "test_ud", "test_2014", "test_2015", "test_2017"]
data = Data(); EV = {r: data.load_eval(r) for r in roles}
tab = np.load(os.path.join(RES, "exp1_instance_table_taskB.npz"))
probe_s = data.sample_train(np.random.default_rng(777), idx=np.random.default_rng(778).choice(data.N, 3851, replace=False))
probe = dict(ctx=probe_s["ctx"], cands=probe_s["cands"], label=probe_s["label"])

print("== grid on val_id NLL only (seed 0) ==")
grid = []
# DISCLOSURE: a first 4-config grid (hidden 256, wd 1e-2, dropout {0.1,0.3} x lr {3e-4,1e-3}) was run first: best val_id NLL 1.763 (worse than the uniform 1.386),
# best epoch 1-3, i.e. the selected point sat on the lr boundary and the model overfit from the start. The grid was then extended (below, val_id only) to lower lr,
# larger weight decay and a narrower bottleneck. No shifted-year data was involved in either step.
for h_, dr, lr, wd in itertools.product([64, 256], [0.1, 0.3], [1e-4, 3e-4, 1e-3], [1e-2, 1e-1]):
    cfg = dict(variant="linear", hidden=h_, dropout=dr, lr=lr, wd=wd, bs=256)
    m, info, _ = train_run(data, cfg, 0, EV["val_id"], probe, os.path.join(OUT, f"grid_h{h_}_do{dr}_lr{lr}_wd{wd}"), verbose=False)
    grid.append((info["val"]["nll"], cfg, info)); print(f"hidden {h_} dropout {dr} lr {lr} wd {wd}: val_id acc {info['val']['acc']*100:.1f} NLL {info['val']['nll']:.4f} best_epoch {info['best_epoch']} params {info['params']:,}", flush=True)
best = min(grid, key=lambda g: g[0]); cfg = best[1]; print("selected:", cfg)
json.dump(dict(selected=cfg, grid=[(g[0], g[1]) for g in grid]), open(os.path.join(OUT, "selected_config.json"), "w"), indent=1)

accs = {r: [] for r in roles}; L = {}
for seed in range(5):
    m, info, _ = train_run(data, cfg, seed, EV["val_id"], probe, os.path.join(OUT, f"seed{seed}"), verbose=False)
    pr = {r: predict(m, data, EV[r]) for r in roles}
    np.savez(os.path.join(OUT, f"seed{seed}", "preds.npz"), **pr)
    for r in roles:
        accs[r].append((pr[r].argmax(1) == EV[r]["label"]).astype(float))
    print(f"seed {seed}: best_epoch {info['best_epoch']} params {info['params']:,} train {info['train_seconds']}s | " + " ".join(f"{r} {accs[r][-1].mean()*100:.1f}" for r in roles), flush=True)

# MLP (existing frozen) and cosine per-instance correctness
mlp = {r: [] for r in roles}
for seed in range(5):
    z = np.load(os.path.join(ROOT, "runs", "phase2", f"visual_seed{seed}", "preds.npz"))
    for r in roles:
        mlp[r].append((z[r].argmax(1) == EV[r]["label"]).astype(float))
cos = {r: tab[f"{r}/acc_cosine"].astype(float) for r in roles}
lin = {r: np.mean(accs[r], 0) for r in roles}; mlpm = {r: np.mean(mlp[r], 0) for r in roles}     # per-instance mean correctness over 5 seeds
users = {r: EV[r]["user"] for r in roles}

print("\n== accuracy (%, mean of 5 seeds; user-cluster 95% CI; chance 25) ==")
print(f"{'set':<10}{'cosine (training-free)':>26}{'linear (5 seeds)':>24}{'MLP (5 seeds)':>24}")
W = {r: boot_weights(users[r], 2000, 777 + i) for i, r in enumerate(roles)}
def bs(r, v):
    return (W[r] @ v) / W[r].sum(1)
for r in roles:
    print(f"{r:<10}" + "".join(f"{v[r].mean()*100:>12.1f} [{ci(bs(r, v[r]))[0]*100:.1f},{ci(bs(r, v[r]))[1]*100:.1f}]" for v in (cos, lin, mlpm)))
print("per-seed linear acc range per set: " + "; ".join(f"{r} {min(a.mean() for a in accs[r])*100:.1f}-{max(a.mean() for a in accs[r])*100:.1f}" for r in roles))

print("\n== DESCRIPTIVE baseline-relative degradation: accuracy change (shifted year - reference), percentage points [independent user-cluster 95% CI] ==")
print("(descriptive only: not additive, not a causal decomposition, not a measure of task difficulty; the three models differ in capacity and in what they exploit)")
Wr = {r: W[r] for r in roles}
for ref in ("test_id", "test_ud"):
    print(f"-- reference {ref}")
    print(f"{'year':<10}{'cosine':>24}{'linear':>24}{'MLP':>24}{'MLP drop - linear drop':>28}")
    for y in ("test_2014", "test_2015", "test_2017"):
        cells = []; d = {}
        for nm, v in (("cos", cos), ("lin", lin), ("mlp", mlpm)):
            dd = bs(y, v[y]) - bs(ref, v[ref]); d[nm] = dd
            cells.append(f"{(v[y].mean()-v[ref].mean())*100:>+9.1f} [{ci(dd)[0]*100:+.1f},{ci(dd)[1]*100:+.1f}]")
        # NB: boot weights of a given set are shared across models -> paired difference of drops is valid
        dm = d["mlp"] - d["lin"]
        pt = ((mlpm[y].mean() - mlpm[ref].mean()) - (lin[y].mean() - lin[ref].mean())) * 100
        print(f"{y:<10}" + "".join(f"{c:>24}" for c in cells) + f"{pt:>+12.1f} [{ci(dm)[0]*100:+.1f},{ci(dm)[1]*100:+.1f}]")
print("\nparameters: cosine 0 (training-free) | linear", f"{best[2]['params']:,}", "| MLP 1,245,953")
