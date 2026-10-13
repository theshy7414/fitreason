"""Experiment 2 step 1: from the FROZEN phase-2 visual checkpoints (seeds 0-4) compute logits and the kNN pair-uncertainty distances. No training.
 bank = scorer hidden activations (256-d, L2-normalised) of (train query, TRUE candidate) pairs, one fixed draw per train outfit (rng 4242);
 query = same activation for (query, model's top-1 predicted candidate). Score = mean Euclidean distance to the k nearest bank vectors, k in {10,50,200}.
Run: PYTHONPATH=./pylibs:./scripts python3 -u scripts/60_extract_scores.py | tee results/exp2_extract_log.txt
"""
import json, time
import numpy as np, torch
from taskb_lib import *

OUT = os.path.join(ROOT, "runs", "exp2"); os.makedirs(OUT, exist_ok=True)
roles = ["val_id", "test_id", "test_ud", "test_2014", "test_2015", "test_2017"]
KS = (10, 50, 200)
data = Data()
EV = {r: data.load_eval(r) for r in roles}
sel = json.load(open(os.path.join(ROOT, "runs", "phase1", "selected_config.json")))["selected"]


@torch.no_grad()
def hidden(model, x):
    ctx_x, mask, cand_x, extra = x
    e = model.enc(cand_x.float()); ec = model.enc(ctx_x.float()) * mask[..., None]
    c = ec.sum(1) / mask.sum(1, keepdim=True).clamp(min=1); c4 = c[:, None].expand_as(e)
    h = model.scorer[1](model.scorer[0](torch.cat([c4, e, c4 * e], -1)))
    return h, model.scorer[3](model.scorer[2](h)).squeeze(-1)


@torch.no_grad()
def run(model, ev, bs=1024):
    H_, L_ = [], []
    for a in range(0, len(ev["label"]), bs):
        h, lg = hidden(model, batch_tensors(data, ev["ctx"][a:a + bs], ev["cands"][a:a + bs], "visual"))
        H_.append(h); L_.append(lg)
    return torch.cat(H_), torch.cat(L_)


@torch.no_grad()
def knn(q, bank, bs=1024):
    out = []
    for a in range(0, len(q), bs):
        sims = q[a:a + bs] @ bank.T
        top = sims.topk(max(KS), dim=1).values
        d = torch.sqrt(torch.clamp(2 - 2 * top, min=0))
        out.append(torch.stack([d[:, :k].mean(1) for k in KS], 1))
    return torch.cat(out)


S = data.sample_train(np.random.default_rng(4242))      # fixed bank draw: 1 instance per train outfit
out, info = {}, {}
for seed in range(5):
    d = os.path.join(ROOT, "runs", "phase2", f"visual_seed{seed}")
    ck = torch.load(os.path.join(d, "best.pt")); cfg = ck["cfg"]; assert cfg["variant"] == "visual"
    model = Net("visual", cfg["hidden"], cfg["dropout"]).to(DEV); model.load_state_dict(ck["state"]); model.eval()
    stored = np.load(os.path.join(d, "preds.npz"))
    # bank
    bh = []
    for a in range(0, data.N, 1024):
        h, _ = hidden(model, batch_tensors(data, S["ctx"][a:a + 1024], S["cands"][a:a + 1024], "visual"))
        bh.append(h[torch.arange(len(h)), torch.from_numpy(S["label"][a:a + 1024]).to(DEV)])
    bank = Fn.normalize(torch.cat(bh), dim=1)
    for r in roles:
        h, lg = run(model, EV[r])
        diff = np.abs(lg.cpu().numpy() - stored[r]).max()
        assert diff < 2e-3, f"recomputed logits differ from stored preds ({diff})"
        pred = lg.argmax(1)
        q = Fn.normalize(h[torch.arange(len(h)), pred], dim=1)
        dk = knn(q, bank).cpu().numpy()
        out[f"s{seed}/{r}/logits"] = lg.cpu().numpy()
        for j, k in enumerate(KS):
            out[f"s{seed}/{r}/knn{k}"] = dk[:, j]
    print(f"seed {seed}: bank {tuple(bank.shape)}; logits match stored preds; mean knn10 per set: " +
          " ".join(f"{r}={out[f's{seed}/{r}/knn10'].mean():.3f}" for r in roles), flush=True)
np.savez(os.path.join(OUT, "scores.npz"), **out)

# ---------------------------------------------------------------- cost measurements (seed 0, test_2015, 7,702 instances)
ev = EV["test_2015"]; n = len(ev["label"]); reps = 10
models = []
for seed in range(5):
    ck = torch.load(os.path.join(ROOT, "runs", "phase2", f"visual_seed{seed}", "best.pt"))
    m = Net("visual", ck["cfg"]["hidden"], ck["cfg"]["dropout"]).to(DEV); m.load_state_dict(ck["state"]); m.eval(); models.append(m)


def timeit(fn):
    fn(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize(); return (time.time() - t0) / reps


t_single = timeit(lambda: predict(models[0], data, ev))
t_ens = timeit(lambda: [predict(m, data, ev) for m in models])
bank = Fn.normalize(torch.randn(data.N, 256, device=DEV), dim=1)
t_hid = timeit(lambda: run(models[0], ev))
hh, _ = run(models[0], ev); q = Fn.normalize(hh[:, 0], dim=1)
torch.cuda.reset_peak_memory_stats(); t_knn = timeit(lambda: knn(q, bank)); peak = torch.cuda.max_memory_allocated() / 1e6
t_bank = timeit(lambda: [hidden(models[0], batch_tensors(data, S["ctx"][a:a + 1024], S["cands"][a:a + 1024], "visual")) for a in range(0, data.N, 1024)])
tr = [json.load(open(os.path.join(ROOT, "runs", "phase2", f"visual_seed{s}", "run_info.json"))) for s in range(5)]
cost = dict(n_instances=n, single_forward_s=t_single, ensemble5_forward_s=t_ens, single_forward_with_hidden_s=t_hid, knn_search_s=t_knn,
            knn_total_s=t_hid + t_knn, bank_build_s=t_bank, bank_bytes=int(data.N * 256 * 4), knn_peak_gpu_MB_incl_data=peak,
            params_single=tr[0]["params"], params_ensemble=sum(t["params"] for t in tr), train_seconds_single=tr[0]["train_seconds"],
            train_seconds_ensemble=sum(t["train_seconds"] for t in tr), env=env_info())
json.dump(cost, open(os.path.join(OUT, "cost.json"), "w"), indent=1)
print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in cost.items() if k != "env"}, indent=1))
