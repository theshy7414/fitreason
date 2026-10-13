"""Experiment 2 statistics helpers: weighted (user-cluster bootstrap) AUROC / risk / ECE with tie handling. Positive class = ERROR; u = uncertainty."""
import zlib
import numpy as np


def boot_weights(users, B, seed):
    uu, inv = np.unique(users, return_inverse=True)
    rs = np.random.RandomState(seed)
    idx = rs.randint(len(uu), size=(B, len(uu)))
    cnt = np.zeros((B, len(uu)), np.int32)
    for b in range(B):
        cnt[b] = np.bincount(idx[b], minlength=len(uu))
    return cnt[:, inv]


def seed_of(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) % (2 ** 31)


class Scorer:
    def __init__(self, u, err):
        self.ug, self.inv = np.unique(u, return_inverse=True)
        self.err = np.asarray(err, float); self.k = len(self.ug)

    def auc(self, w):
        wp = np.bincount(self.inv, weights=w * self.err, minlength=self.k)
        wn = np.bincount(self.inv, weights=w * (1 - self.err), minlength=self.k)
        P, N = wp.sum(), wn.sum()
        if P == 0 or N == 0:
            return np.nan
        return float((wp * (np.cumsum(wn) - wn + 0.5 * wn)).sum() / (P * N))

    def _cum(self, w):
        Wg = np.bincount(self.inv, weights=w, minlength=self.k); Eg = np.bincount(self.inv, weights=w * self.err, minlength=self.k)
        return Wg, Eg, np.cumsum(Wg), np.cumsum(Eg)

    def risk_cov(self, w, cov=0.8):
        Wg, Eg, cW, cE = self._cum(w); tgt = cov * cW[-1]
        j = min(int(np.searchsorted(cW, tgt - 1e-9)), self.k - 1)
        pW, pE = cW[j] - Wg[j], cE[j] - Eg[j]
        fr = min(max((tgt - pW) / Wg[j], 0.0), 1.0) if Wg[j] > 0 else 0.0
        den = pW + fr * Wg[j]
        return float((pE + fr * Eg[j]) / den) if den > 0 else np.nan

    def curve(self, w, covs):
        Wg, Eg, cW, cE = self._cum(w); out = []
        for cov in covs:
            tgt = cov * cW[-1]; j = min(int(np.searchsorted(cW, tgt - 1e-9)), self.k - 1)
            pW, pE = cW[j] - Wg[j], cE[j] - Eg[j]
            fr = min(max((tgt - pW) / Wg[j], 0.0), 1.0) if Wg[j] > 0 else 0.0
            out.append((pE + fr * Eg[j]) / (pW + fr * Wg[j]))
        return np.array(out)

    def fixed(self, w, acc):
        wa = w * acc; s = wa.sum()
        return float(s / w.sum()), (float((wa * self.err).sum() / s) if s > 0 else np.nan), float((wa * self.err).sum() / w.sum())


class Ece:
    def __init__(self, conf, ok, bins=15):
        self.o = np.argsort(conf, kind="mergesort"); self.c = conf[self.o]; self.ok = np.asarray(ok, float)[self.o]; self.bins = bins

    def __call__(self, w):
        w = w[self.o]; cw = np.cumsum(w); W = cw[-1]
        b = np.minimum(((cw - 0.5 * w) / W * self.bins).astype(int), self.bins - 1)
        return float(np.abs(np.bincount(b, weights=w * (self.ok - self.c), minlength=self.bins)).sum() / W)


def ci(a, lo=2.5, hi=97.5):
    a = np.asarray(a); a = a[~np.isnan(a)]
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


def cap_mask(users, cap, seed):
    rs = np.random.RandomState(seed); perm = rs.permutation(len(users)); seen = {}; keep = np.zeros(len(users), bool)
    for i in perm:
        c = seen.get(users[i], 0)
        if c < cap:
            keep[i] = True; seen[users[i]] = c + 1
    return keep
