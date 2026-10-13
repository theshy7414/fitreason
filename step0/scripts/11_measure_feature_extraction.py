"""Step 0: measure real feature-decoding cost on ONE tar (11,678 items) and check item coverage vs. an example experiment.
Run: PYTHONPATH=./pylibs python3 scripts/11_measure_feature_extraction.py
"""
import gzip, json, os, tarfile, time
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
Z = np.load(os.path.join(ROOT, "data", "outfits", "outfits_compact.npz"))
year, like, n_items, item_id, cat1 = Z["year"], Z["like"], Z["n_items"], Z["item_id"], Z["cat1"]
N = len(year)
off = np.zeros(N + 1, dtype=np.int64); np.cumsum(n_items, out=off[1:])
outfit_of_slot = np.repeat(np.arange(N), n_items)
bit = 1 << (cat1 - cat1.min()).astype(np.int64)
mask = np.bitwise_or.reduceat(bit, off[:-1])
popc = np.array([bin(i).count("1") for i in range(128)])
elig = (popc[mask] >= 4) & (like >= 50)

# items needed for a 2013 -> 2014 official-size experiment
rs = np.random.RandomState(0)
tr = rs.permutation(np.where(elig & (year == 2013))[0])[:30816]
te = rs.permutation(np.where(elig & (year == 2014))[0])[:7702]
sel = np.concatenate([tr, te])
need = np.unique(item_id[np.concatenate([np.arange(off[o], off[o + 1]) for o in sel])])
need_set = set(need.tolist())
print("items needed for 2013->2014 experiment:", len(need))

tpath = os.path.join(ROOT, "data", "features_tar_sample", "sample.tar.gz")
t0 = time.time()
ids_all, ids_need, vecs = [], [], []
with tarfile.open(tpath, "r:gz") as tf:
    for m in tf:
        if not m.isfile():
            continue
        iid = int(os.path.basename(m.name).split(".")[0])
        ids_all.append(iid)
        if iid in need_set:
            v = np.array(json.loads(gzip.decompress(tf.extractfile(m).read())), dtype=np.float16)
            ids_need.append(iid); vecs.append(v)
dt = time.time() - t0
print(f"tar scanned+filtered in {dt:.1f}s | files in tar={len(ids_all)} | needed items found in this tar={len(ids_need)} "
      f"({len(ids_need)/len(need)*100:.2f}% of needed; expected ~{100/200:.2f}% if hash-partitioned over 200 tars)")
# decode-all timing on a subset to extrapolate worst case
t1 = time.time(); k = 0
with tarfile.open(tpath, "r:gz") as tf:
    for m in tf:
        if m.isfile():
            _ = np.array(json.loads(gzip.decompress(tf.extractfile(m).read())), dtype=np.float16); k += 1
            if k >= 2000: break
per = (time.time() - t1) / k
print(f"decode-all timing: {per*1000:.2f} ms/file -> full 2.34M items single-thread ~ {per*2335598/3600:.2f} h")
arr = np.stack(vecs)
print("stacked sample:", arr.shape, arr.dtype, "| fp16 finite:", bool(np.isfinite(arr).all()),
      "| max abs:", float(np.abs(arr.astype(np.float32)).max()))
print("fraction of fp16-overflow (>65504):", float((np.abs(arr.astype(np.float32)) > 65000).mean()))
