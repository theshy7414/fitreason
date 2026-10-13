"""Task B training library: frozen VGG16 features -> permutation-invariant (mean-pool) outfit representation -> small scoring MLP.
Uses ONLY the frozen S1 train-2013 manifests. Training negatives are sampled online with the SAME rule as the frozen eval instances
(same category_id1, other user, not an item of the query outfit, distinct). Eval instances are read from the manifests, never regenerated.
"""
import os, json, time, hashlib, platform, subprocess, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from exp1_common import *

DEV = "cuda"
CATS = list(range(10, 17))
NC = len(CATS)


class Data:
    def __init__(self):
        F, rid = load_required_features()
        self.rid = rid
        R = roles_from_outfits("S1_tr2013_outfits.jsonl.gz")
        self.train_o = R["train"]
        tr_items = items_in(self.train_o)
        Ftr = F[np.searchsorted(rid, tr_items)].astype(np.float32)
        self.mu, self.sd = Ftr.mean(0), Ftr.std(0) + 1e-6          # statistics from TRAIN items only
        self.min_sd = float(Ftr.std(0).min()); del Ftr
        self.X = torch.empty((len(rid), 4096), dtype=torch.float16, device=DEV)
        mu_t, sd_t = torch.from_numpy(self.mu).to(DEV), torch.from_numpy(self.sd).to(DEV)
        mx = 0.0
        for a in range(0, len(rid), 50000):
            z = (torch.from_numpy(F[a:a + 50000]).to(DEV).float() - mu_t) / sd_t
            mx = max(mx, float(z.abs().max())); self.X[a:a + 50000] = z.half()
        self.max_abs_z = mx
        self.cat_row = torch.from_numpy(np.array([item_cat1[int(i)] for i in rid]) - CATS[0]).to(DEV)
        self._build_train()

    def row(self, items):
        items = np.asarray(items); r = np.searchsorted(self.rid, items)
        assert (self.rid[np.minimum(r, len(self.rid) - 1)] == items).all(), "item without a feature row"
        return r

    def _build_train(self):
        O = self.train_o; N = len(O); L = int(n_items[O].max())
        self.N, self.L = N, L
        self.items_pad = -np.ones((N, L), np.int64); self.cats_pad = -np.ones((N, L), np.int64)
        self.nit = n_items[O].astype(np.int64); self.tuser = user_id[O].astype(np.int64)
        pools = {c: ([], [], []) for c in CATS}
        for i, o in enumerate(O):
            its = self.row(items_of(o)); n = len(its)
            self.items_pad[i, :n] = its; self.cats_pad[i, :n] = cat1[off[o]:off[o + 1]]
            for j in range(n):
                p = pools[int(cat1[off[o] + j])]; p[0].append(its[j]); p[1].append(i); p[2].append(self.tuser[i])
        self.pools = {c: tuple(np.array(a, np.int64) for a in p) for c, p in pools.items()}

    # ------------------------------------------------------------------ online training instances
    def sample_train(self, rng, idx=None):
        idx = np.arange(self.N) if idx is None else np.asarray(idx)
        n = len(idx); ip = self.items_pad[idx]; cp = self.cats_pad[idx]; nit = self.nit[idx]; us = self.tuser[idx]
        k = (rng.random(n) * nit).astype(np.int64); ar = np.arange(n)
        pos = ip[ar, k]; pc = cp[ar, k]
        ctx = ip.copy(); ctx[ar, k] = -1
        negs = np.zeros((n, 3), np.int64)
        for c in CATS:
            R = np.where(pc == c)[0]
            if len(R) == 0:
                continue
            pr, po, pu = self.pools[c]
            todo = R
            for _ in range(50):
                if len(todo) == 0:
                    break
                dr = rng.integers(len(pr), size=(len(todo), 24))
                ci, ow, ur = pr[dr], po[dr], pu[dr]
                valid = (ow != idx[todo][:, None]) & (ur != us[todo][:, None]) & ~(ci[:, :, None] == ip[todo][:, None, :]).any(-1)
                fail = []
                for t in range(len(todo)):
                    ch = []
                    for j in np.where(valid[t])[0]:
                        if ci[t, j] not in ch:
                            ch.append(ci[t, j])
                            if len(ch) == 3:
                                break
                    if len(ch) == 3:
                        negs[todo[t]] = ch
                    else:
                        fail.append(t)
                todo = todo[fail]
            assert len(todo) == 0, "could not sample negatives"
        cands = np.concatenate([pos[:, None], negs], 1)
        perm = rng.random((n, 4)).argsort(1)
        cands = np.take_along_axis(cands, perm, 1); label = np.where(perm == 0)[1]
        return dict(ctx=ctx, cands=cands, label=label.astype(np.int64), user=us, cat=pc - CATS[0], pos=pos)

    # ------------------------------------------------------------------ frozen eval instances
    def load_eval(self, role):
        rows = rd(f"S1_tr2013_taskB_{role}.jsonl.gz")
        L = max(len(r["ctx"]) for r in rows); n = len(rows)
        ctx = -np.ones((n, L), np.int64); cands = np.zeros((n, 4), np.int64)
        for i, r in enumerate(rows):
            ctx[i, :len(r["ctx"])] = self.row(r["ctx"]); cands[i] = self.row(r["cands"])
        return dict(ctx=ctx, cands=cands, label=np.array([r["label"] for r in rows]), user=np.array([r["user_id"] for r in rows]),
                    cat=np.array([r["cat1"] for r in rows]) - CATS[0], ids=[r["instance_id"] for r in rows], role=role)


# ============================================================================== model
class Net(nn.Module):
    def __init__(self, variant="visual", h=256, p=0.1):
        super().__init__()
        self.variant = variant
        if variant == "linear":      # supplementary descriptive baseline (Exp.2 round): linear encoder, bilinear dot-product score + linear candidate term; no nonlinearity
            self.enc = nn.Sequential(nn.Dropout(p), nn.Linear(4096, h)); self.cand_lin = nn.Linear(h, 1, bias=False)
            return
        self.enc = nn.Sequential(nn.Dropout(p), nn.Linear(4096, h), nn.GELU())
        extra = (NC + NC + 1) if variant == "visual_cat" else 0
        din = h if variant == "cand_only" else 3 * h + extra
        self.scorer = nn.Sequential(nn.Linear(din, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, 1))

    def forward(self, ctx_x, mask, cand_x, extra=None):
        e = self.enc(cand_x.float())                                   # [B,4,h]
        if self.variant == "linear":
            ec = self.enc(ctx_x.float()) * mask[..., None]
            c = ec.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            return (c[:, None] * e).sum(-1) + self.cand_lin(e).squeeze(-1)
        if self.variant == "cand_only":
            return self.scorer(e).squeeze(-1)
        ec = self.enc(ctx_x.float()) * mask[..., None]                 # [B,L,h]
        c = ec.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)         # mean pooling: permutation invariant
        c4 = c[:, None].expand_as(e)
        z = [c4, e, c4 * e]
        if extra is not None:
            z.append(extra)
        return self.scorer(torch.cat(z, -1)).squeeze(-1)


def batch_tensors(data, ctx_idx, cand_idx, variant, zero_ctx=False):
    ctx_idx = torch.as_tensor(ctx_idx, device=DEV); cand_idx = torch.as_tensor(cand_idx, device=DEV)
    mask = (ctx_idx >= 0).float()
    if zero_ctx:
        mask = mask * 0
    ci = ctx_idx.clamp(min=0)
    ctx_x = data.X[ci]; cand_x = data.X[cand_idx]
    extra = None
    if variant == "visual_cat":
        oh = Fn.one_hot(data.cat_row[ci], NC).float() * mask[..., None]
        n = mask.sum(1, keepdim=True).clamp(min=1)
        hist = oh.sum(1) / n                                           # [B,7]
        cc = Fn.one_hot(data.cat_row[cand_idx], NC).float()            # [B,4,7]
        extra = torch.cat([hist[:, None].expand(-1, 4, -1), cc, torch.log(n)[:, None].expand(-1, 4, 1)], -1)
    return ctx_x, mask, cand_x, extra


@torch.no_grad()
def predict(model, data, ev, bs=512, ctx_override=None, zero_ctx=False):
    model.eval(); out = []
    ctx = ev["ctx"] if ctx_override is None else ctx_override
    for a in range(0, len(ev["label"]), bs):
        x = batch_tensors(data, ctx[a:a + bs], ev["cands"][a:a + bs], model.variant, zero_ctx)
        out.append(model(*x).float().cpu().numpy())
    return np.concatenate(out)


def metrics(logits, label):
    lp = logits - logits.max(1, keepdims=True); lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
    return dict(acc=float((logits.argmax(1) == label).mean()), nll=float(-lp[np.arange(len(label)), label].mean()))


def softmax(l):
    e = np.exp(l - l.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def train_run(data, cfg, seed, ev_val, probe, outdir, max_epochs=40, patience=5, shuffled_labels=False, verbose=True):
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    model = Net(cfg["variant"], cfg["hidden"], cfg["dropout"]).to(DEV)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    best, best_ep, bad, log = 1e9, -1, 0, []
    t_train = 0.0
    for ep in range(1, max_epochs + 1):
        t0 = time.time(); model.train()
        S = data.sample_train(rng)
        if shuffled_labels:
            S["label"] = rng.integers(4, size=len(S["label"]))
        perm = rng.permutation(data.N); tl = ta = 0.0
        for a in range(0, data.N, cfg["bs"]):
            b = perm[a:a + cfg["bs"]]
            x = batch_tensors(data, S["ctx"][b], S["cands"][b], cfg["variant"])
            y = torch.from_numpy(S["label"][b]).to(DEV)
            lg = model(*x); loss = Fn.cross_entropy(lg, y)
            assert torch.isfinite(loss), f"non-finite loss at epoch {ep}"
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tl += float(loss) * len(b); ta += float((lg.argmax(1) == y).float().sum())
        torch.cuda.synchronize(); t_train += time.time() - t0
        mv = metrics(predict(model, data, ev_val), ev_val["label"]); mp = metrics(predict(model, data, probe), probe["label"])
        row = dict(epoch=ep, train_loss=tl / data.N, train_acc_online=ta / data.N, probe_acc=mp["acc"], probe_nll=mp["nll"],
                   val_acc=mv["acc"], val_nll=mv["nll"], sec=round(time.time() - t0, 2))
        log.append(row)
        if verbose:
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        if mv["nll"] < best - 1e-5:
            best, best_ep, bad = mv["nll"], ep, 0
            torch.save(dict(state=model.state_dict(), cfg=cfg, seed=seed, epoch=ep, val=mv), os.path.join(outdir, "best.pt"))
        else:
            bad += 1
            if bad >= patience:
                break
    with open(os.path.join(outdir, "log.jsonl"), "w") as f:
        for r in log:
            f.write(json.dumps(r) + "\n")
    ck = torch.load(os.path.join(outdir, "best.pt")); model.load_state_dict(ck["state"])
    info = dict(cfg=cfg, seed=seed, best_epoch=best_ep, epochs_run=len(log), params=nparams, train_seconds=round(t_train, 1),
                sec_per_epoch=round(t_train / len(log), 2), val=ck["val"], shuffled_labels=shuffled_labels)
    json.dump(info, open(os.path.join(outdir, "run_info.json"), "w"), indent=1)
    return model, info, log


def env_info():
    return dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                platform=platform.platform(), manifest_sha256=hashlib.sha256(open(os.path.join(MAN, "MANIFEST_SHA256.txt"), "rb").read()).hexdigest())
