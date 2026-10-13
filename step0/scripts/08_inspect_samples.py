"""Step 0: inspect the small samples (item catalog, year labels, one VGG16 feature tar).
Run:  PYTHONPATH=./pylibs python3 scripts/08_inspect_samples.py
"""
import gzip, json, os, tarfile, time, glob, collections
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
D = os.path.join(ROOT, "data")

print("=== item_catalog.txt ===")
path = os.path.join(D, "catalog", "item_catalog.txt")
with open(path, encoding="utf-8") as f:
    head = [next(f) for _ in range(5)]
for l in head:
    print(repr(l[:200]))
df = pd.read_csv(path, sep="\t", header=None, nrows=None)
print("shape", df.shape)
print(df.head(5).to_string())
print(df.dtypes)

print("\n=== year_classification_labels ===")
for p in sorted(glob.glob(os.path.join(D, "year_labels", "year_classification_labels", "*")))[:15]:
    print(p.replace(D, "data"), os.path.getsize(p) if os.path.isfile(p) else "dir")
sub = glob.glob(os.path.join(D, "year_labels", "year_classification_labels", "*", "*"))[:8]
for p in sub:
    print(" ", p.replace(D, "data"), os.path.getsize(p))

print("\n=== sample feature tar ===")
tpath = os.path.join(D, "features_tar_sample", "sample.tar.gz")
t0 = time.time()
n = 0
first = None
names = []
with tarfile.open(tpath, "r:gz") as tf:
    for m in tf:
        if m.isfile():
            n += 1
            if len(names) < 3:
                names.append(m.name)
            if first is None:
                first = (m.name, tf.extractfile(m).read())
print("members:", n, "scan seconds:", round(time.time() - t0, 1))
print("first names:", names)
raw = gzip.decompress(first[1])
vec = np.array(json.loads(raw), dtype=np.float32)
print("first file:", first[0], "-> decoded shape", vec.shape, "dtype", vec.dtype,
      "min/max/mean", float(vec.min()), float(vec.max()), round(float(vec.mean()), 3),
      "frac<0", round(float((vec < 0).mean()), 3), "frac==0", round(float((vec == 0).mean()), 3))
print("raw json bytes per feature file:", len(raw), "gz bytes:", len(first[1]))
