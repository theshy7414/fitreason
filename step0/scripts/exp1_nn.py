"""Shared nearest-neighbour helpers (GPU, within category_id1, train-mean-centred cosine, identical item_id excluded)."""
import collections
import numpy as np, torch
from exp1_common import *

dev = "cuda"
F, rid = load_required_features()


def vec(items):
    return F[np.searchsorted(rid, items)]


def normed(items, mu):
    x = vec(items).astype(np.float32) - mu
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x


def topk_sims(Ev, e_ids, Tt, tid, k=10, chunk=2048):
    outv, outi = [], []
    for s in range(0, len(Ev), chunk):
        E = torch.from_numpy(Ev[s:s + chunk]).to(dev)
        S = (E @ Tt.T).float()
        eid = torch.from_numpy(e_ids[s:s + chunk]).to(dev)
        S[eid[:, None] == tid[None, :]] = -2.0
        v, i = S.topk(min(k, S.shape[1]), dim=1)
        outv.append(v.cpu().numpy()); outi.append(tid[i].cpu().numpy())
    return np.concatenate(outv), np.concatenate(outi)


def nn_tables(train_o, eval_outfits):
    T = items_in(train_o); U = items_in(eval_outfits)
    mu = vec(T).astype(np.float32).mean(0)
    nn_s = np.full((len(U), 10), -2.0, np.float32); nn_i = np.full((len(U), 10), -1, np.int64)
    ref_top1 = collections.defaultdict(list)
    t_cat = np.array([item_cat1[i] for i in T]); u_cat = np.array([item_cat1[i] for i in U])
    rs = np.random.RandomState(0)
    for c in np.unique(u_cat):
        Tc = T[t_cat == c]; Uc_pos = np.where(u_cat == c)[0]
        Tt = torch.from_numpy(normed(Tc, mu)).half().to(dev); tid = torch.from_numpy(Tc).to(dev)
        v, i = topk_sims(normed(U[Uc_pos], mu).astype(np.float16), U[Uc_pos], Tt, tid)
        nn_s[Uc_pos, :v.shape[1]] = v; nn_i[Uc_pos, :v.shape[1]] = i
        sub = rs.choice(len(Tc), min(3000, len(Tc)), replace=False)
        v2, _ = topk_sims(normed(Tc[sub], mu).astype(np.float16), Tc[sub], Tt, tid, k=1)
        ref_top1[int(c)] = v2[:, 0]
        del Tt
    return T, U, mu, nn_s, nn_i, ref_top1
