"""Post-extraction verification: per-tar stats, memmap ordering vs shards, coverage of every frozen manifest set, feature scale."""
import json, os, gzip
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "data", "features_v1")
d = [json.loads(l) for l in open(os.path.join(OUT, "done.jsonl"))]
print("tars:", len(d), "| all size+ETag verified:", all(x["size_ok"] and x["etag_ok"] for x in d), "| dir name matches tar:", all(x["dir_matches_tar_name"] for x in d))
print("files in tars:", sum(x["n_in_tar"] for x in d), "| kept:", sum(x["n_kept"] for x in d), "| bad_dim:", sum(x["bad_dim"] for x in d),
      "| nonfinite:", sum(x["nonfinite"] for x in d), "| fp16 overflow:", sum(x["overflow"] for x in d))
ids = np.load(os.path.join(OUT, "item_ids.npy")); F = np.load(os.path.join(OUT, "features_fp16.npy"), mmap_mode="r")
print("matrix:", F.shape, F.dtype, "| ids sorted & unique:", bool((np.diff(ids) > 0).all()))
# ordering check: 3 shards, compare 50 random rows
rs = np.random.RandomState(0)
shards = sorted(os.listdir(os.path.join(OUT, "shards")))
for s in [shards[0], shards[100], shards[-1]]:
    z = np.load(os.path.join(OUT, "shards", s))
    for j in rs.choice(len(z["ids"]), 50, replace=False):
        r = np.searchsorted(ids, z["ids"][j]); assert ids[r] == z["ids"][j] and np.array_equal(F[r], z["feats"][j])
print("memmap rows match shard rows for 150 random items: OK")
samp = np.asarray(F[np.sort(rs.choice(len(ids), 20000, replace=False))]).astype(np.float32)
print(f"feature scale (20k sample): mean={samp.mean():.3f} std={samp.std():.3f} max={samp.max():.1f} min={samp.min():.1f} zero-rows={int((np.abs(samp).sum(1)==0).sum())}")
# catalog agreement
cat = np.loadtxt(os.path.join(ROOT, "data/catalog/item_catalog.txt"), dtype=np.int64)
print("catalog items:", len(cat), "| all have features:", bool(np.isin(cat[:, 0], ids).all()))
# coverage of every frozen manifest set
req = np.array(open(os.path.join(ROOT, "manifests/required_items_v1.txt")).read().split(), dtype=np.int64)
print("required items:", len(req), "| covered:", int(np.isin(req, ids).sum()))
