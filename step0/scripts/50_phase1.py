"""Phase 1: minimal Task B MLP validation (visual-only, single fixed seed 0). Only train + val_id are used for selection/tuning.
Shifted-year sets are NOT loaded. test_id is evaluated once, after the config is frozen.
Run: PYTHONPATH=./pylibs:./scripts python3 -u scripts/50_phase1.py 2>&1 | tee results/exp1_phase1_log.txt
"""
import itertools, json, time
import numpy as np, torch
from taskb_lib import *

OUT = os.path.join(ROOT, "runs", "phase1"); os.makedirs(OUT, exist_ok=True)
SEED = 0
t_all = time.time()
data = Data()
H("0. ENV / DATA")
print(json.dumps(env_info()))
print(f"features on GPU: {tuple(data.X.shape)} fp16 | train-item std min={data.min_sd:.4f} | max |z| over all items={data.max_abs_z:.1f} | "
      f"finite: {bool(torch.isfinite(data.X[::97].float()).all())}")
val = data.load_eval("val_id")

# ----------------------------------------------------------------------------- integrity checks
H("1. INTEGRITY CHECKS")
rs = np.random.RandomState(0)
full_ids = np.load(os.path.join(FEAT, "item_ids.npy")); M = np.load(os.path.join(FEAT, "features_fp16.npy"), mmap_mode="r")
samp = rs.choice(len(data.rid), 400, replace=False)
raw = np.stack([np.asarray(M[np.searchsorted(full_ids, data.rid[i])]) for i in samp]).astype(np.float32)   # independent route: full matrix by item_id
z_ref = (raw - data.mu) / data.sd
z_got = data.X[torch.from_numpy(samp).to(DEV)].float().cpu().numpy()
print(f"(a) feature rows: GPU standardised row vs independent full-matrix lookup by item_id, 400 random items: max|diff|={np.abs(z_ref - z_got).max():.4f}")
assert np.abs(z_ref - z_got).max() < 0.05
# train slot -> row -> item id round trip
bad = 0
for i in rs.choice(data.N, 300, replace=False):
    o = data.train_o[i]; its = items_of(o)
    bad += int((data.rid[data.items_pad[i, :len(its)]] != its).any())
print(f"(b) train items_pad round-trip (row -> item_id == manifest item_id) mismatches: {bad}/300"); assert bad == 0
cr = data.cat_row.cpu().numpy() + CATS[0]
print("    category per row matches item_cat1:", all(cr[i] == item_cat1[int(data.rid[i])] for i in samp)); assert all(cr[i] == item_cat1[int(data.rid[i])] for i in samp)
S = data.sample_train(np.random.default_rng(123))
ok_ctx = ok_dist = ok_cat = ok_user = ok_out = 0
for i in range(0, data.N, 7):
    c = S["cands"][i]; lab = S["label"][i]; pos = c[lab]
    ctxs = set(S["ctx"][i][S["ctx"][i] >= 0].tolist())
    outfit = set(data.items_pad[i][data.items_pad[i] >= 0].tolist())
    negs = [x for j, x in enumerate(c) if j != lab]
    ok_ctx += pos not in ctxs; ok_dist += len(set(c.tolist())) == 4
    ok_cat += all(cr[x] == cr[pos] for x in negs); ok_out += all(x not in outfit for x in negs)
n_chk = len(range(0, data.N, 7))
print(f"(c) online train sampler on {n_chk} outfits: masked item not in ctx {ok_ctx}/{n_chk}; 4 distinct cands {ok_dist}/{n_chk}; negatives same cat1 {ok_cat}/{n_chk}; "
      f"negatives not in query outfit {ok_out}/{n_chk}")
assert min(ok_ctx, ok_dist, ok_cat, ok_out) == n_chk
print(f"    train label positions: {np.bincount(S['label'], minlength=4) / len(S['label'])}")
inv = 0
for role_ev in [val]:
    for i in range(len(role_ev["label"])):
        pos = role_ev["cands"][i, role_ev["label"][i]]
        inv += int(pos in set(role_ev["ctx"][i][role_ev["ctx"][i] >= 0].tolist())) + int(len(set(role_ev["cands"][i].tolist())) != 4)
print(f"(d) val_id frozen instances: masked item in ctx / duplicate candidates: {inv} violations; label positions {np.bincount(val['label'], minlength=4)/len(val['label'])}")
assert inv == 0
probe = data.sample_train(np.random.default_rng(777), idx=np.random.default_rng(778).choice(data.N, 3851, replace=False))
probe = dict(ctx=probe["ctx"], cands=probe["cands"], label=probe["label"])      # fixed train-probe (NOT a manifest set)

# ----------------------------------------------------------------------------- grid on val_id NLL (seed 0 for all)
H("2. SMALL GRID (selection by val_id NLL only; seed 0)")
grid = [dict(variant="visual", hidden=256, dropout=d, lr=lr, wd=1e-2, bs=256) for lr, d in itertools.product([1e-3, 3e-4], [0.1, 0.3])]
res = []
for gi, cfg in enumerate(grid):
    print(f"-- config {gi}: {cfg}", flush=True)
    model, info, log = train_run(data, cfg, SEED, val, probe, os.path.join(OUT, f"grid{gi}"), verbose=False)
    print(f"   best_epoch={info['best_epoch']} epochs_run={info['epochs_run']} val_acc={info['val']['acc']*100:.2f} val_nll={info['val']['nll']:.4f} "
          f"train_s={info['train_seconds']}", flush=True)
    res.append((info["val"]["nll"], gi, model, info))
res.sort(key=lambda t: t[0])
best_nll, gi, model, info = res[0]
cfg = grid[gi]
print(f"\nSELECTED config {gi}: {cfg} (val_nll={best_nll:.4f}, val_acc={info['val']['acc']*100:.2f}%, best_epoch={info['best_epoch']})")
json.dump(dict(selected=cfg, grid_index=gi, seed=SEED, grid_results=[dict(cfg=grid[g], val=i["val"], best_epoch=i["best_epoch"]) for _, g, _, i in res]),
          open(os.path.join(OUT, "selected_config.json"), "w"), indent=1)
log = [json.loads(l) for l in open(os.path.join(OUT, f"grid{gi}", "log.jsonl"))]
print("\nlearning curve of the selected run:\nepoch train_loss train_acc(online,dropout) probe_acc probe_nll val_acc val_nll")
for r in log:
    print(f"{r['epoch']:>5} {r['train_loss']:.4f} {r['train_acc_online']*100:>10.2f} {r['probe_acc']*100:>14.2f} {r['probe_nll']:.4f} {r['val_acc']*100:>8.2f} {r['val_nll']:.4f}")
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6)); e = [r["epoch"] for r in log]
    ax[0].plot(e, [r["train_loss"] for r in log], label="train loss (online, dropout on)"); ax[0].plot(e, [r["probe_nll"] for r in log], label="train-probe NLL (eval mode)")
    ax[0].plot(e, [r["val_nll"] for r in log], label="val_id NLL"); ax[0].axhline(np.log(4), ls=":", c="gray"); ax[0].legend(fontsize=7); ax[0].set_xlabel("epoch")
    ax[1].plot(e, [r["probe_acc"] * 100 for r in log], label="train-probe acc"); ax[1].plot(e, [r["val_acc"] * 100 for r in log], label="val_id acc")
    ax[1].axhline(33.0, ls="--", c="r", label="cosine val_id 33.0"); ax[1].axhline(25, ls=":", c="gray"); ax[1].legend(fontsize=7); ax[1].set_xlabel("epoch")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "learning_curve.png"), dpi=130); print("saved runs/phase1/learning_curve.png")
except Exception as ex:
    print("plot skipped:", ex)

# ----------------------------------------------------------------------------- controls (val_id only)
H("3. CONTROLS on val_id (no test data)")
lg = predict(model, data, val); m0 = metrics(lg, val["label"])
print(f"selected model val_id acc={m0['acc']*100:.2f} nll={m0['nll']:.4f}")
perm = np.random.RandomState(5).permutation(len(val["label"]))
for name, kw in [("context swapped with another instance's context", dict(ctx_override=val["ctx"][perm])), ("context removed (zero mask)", dict(zero_ctx=True))]:
    m = metrics(predict(model, data, val, **kw), val["label"]); print(f"  {name}: acc={m['acc']*100:.2f}  (chance 25.0)")
pred = lg.argmax(1)
print("  predicted-position histogram:", np.round(np.bincount(pred, minlength=4) / len(pred), 3), "| label-position histogram:", np.round(np.bincount(val["label"], minlength=4) / len(pred), 3))
print("  accuracy by label position:", [round(float((pred[val['label'] == k] == k).mean()) * 100, 1) for k in range(4)])
print("-- shuffled-label training control (labels random; 4 epochs)")
_, info_s, log_s = train_run(data, cfg, SEED, val, probe, os.path.join(OUT, "control_shuffled_labels"), max_epochs=4, patience=99, shuffled_labels=True, verbose=False)
print("   val_acc per epoch:", [round(r["val_acc"] * 100, 2) for r in log_s], "(should hover at 25)")
print("-- candidate-only control (no context; same config; full training)")
cfg_c = dict(cfg, variant="cand_only")
mc, info_c, _ = train_run(data, cfg_c, SEED, val, probe, os.path.join(OUT, "control_cand_only"), verbose=False)
print(f"   cand_only val_id acc={info_c['val']['acc']*100:.2f} (should be ~25 if the candidate alone carries no label information) | best_epoch={info_c['best_epoch']}")

# ----------------------------------------------------------------------------- FINAL test_id (once)
H("4. test_id (evaluated once, config frozen)")
te = data.load_eval("test_id")
tab = np.load(os.path.join(RES, "exp1_instance_table_taskB.npz"))
assert list(tab["test_id/instance_id"]) == te["ids"], "instance order mismatch with instance table"
lg_te = predict(model, data, te); correct = (lg_te.argmax(1) == te["label"]).astype(float)
users = te["user"]
acc_ci = cluster_boot(correct, users, B=2000)
print(f"MLP visual-only test_id accuracy: {fmt(acc_ci)}  (n={len(correct)} instances, users={len(set(users))}, effective users={eff_users(users)[0]:.0f})")
print(f"{'baseline':<22}{'acc [95% CI]':>22}{'paired diff MLP - baseline (pp) [95% user-cluster CI]':>62}")
for nm in ["cosine", "price", "cat2prior", "popularity", "cooc"]:
    b = tab[f"test_id/acc_{nm}"]
    print(f"{nm:<22}{fmt(cluster_boot(b, users, B=2000)):>22}{fmt(cluster_boot(correct - b, users, B=2000)):>62}")
m_te = metrics(lg_te, te["label"]); print(f"test_id NLL={m_te['nll']:.4f} (uniform = {np.log(4):.4f})")
np.savez(os.path.join(OUT, "preds_selected.npz"), val_id=lg, test_id=lg_te)

# ----------------------------------------------------------------------------- cost
H("5. COST")
nparams = sum(p.numel() for p in model.parameters())
print(f"parameters: {nparams:,} (encoder {sum(p.numel() for p in model.enc.parameters()):,} + scorer {sum(p.numel() for p in model.scorer.parameters()):,})")
print(f"training: {info['train_seconds']}s for {info['epochs_run']} epochs ({info['sec_per_epoch']}s/epoch) on {torch.cuda.get_device_name(0)}; "
      f"whole phase-1 script wall-clock {time.time()-t_all:.0f}s incl. grid + controls")
torch.cuda.synchronize(); t0 = time.time(); reps = 10
for _ in range(reps):
    predict(model, data, te)
torch.cuda.synchronize(); dt = (time.time() - t0) / reps
print(f"inference GPU: {len(te['label'])/dt:,.0f} instances/s ({dt/len(te['label'])*1e3:.3f} ms/instance, includes feature gather, batch 512)")
mcpu = Net(cfg["variant"], cfg["hidden"], cfg["dropout"]); mcpu.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()}); mcpu.eval()
torch.set_num_threads(4)
x = batch_tensors(data, te["ctx"][:512], te["cands"][:512], "visual"); x = tuple(None if t is None else t.cpu() for t in x)
t0 = time.time()
with torch.no_grad():
    for _ in range(5):
        mcpu(*x)
dt = (time.time() - t0) / 5
print(f"inference CPU (4 threads, features pre-gathered): {512/dt:,.0f} instances/s")
json.dump(dict(selected=cfg, test_id_acc=acc_ci, params=nparams, train_seconds=info["train_seconds"], env=env_info()), open(os.path.join(OUT, "phase1_summary.json"), "w"), indent=1)
