"""Step 0: stream the 2.68GB outfit json once and save a compact, columnar copy (metadata only).
Run:  PYTHONPATH=./pylibs python3 scripts/09_stream_outfits.py
Output: data/outfits/outfits_compact.npz   (gitignored / outside any repo)
"""
import os, sys, time, datetime
import numpy as np
import ijson

ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC = os.path.join(ROOT, "data", "outfits", "iqon_outfits.json")
DST = os.path.join(ROOT, "data", "outfits", "outfits_compact.npz")

print("ijson backend:", ijson.backend, flush=True)

set_id, user_id, like, n_items, year, ymd = [], [], [], [], [], []
item_id, cat1, cat2, price = [], [], [], []
bad = 0
t0 = time.time()
size = os.path.getsize(SRC)
with open(SRC, "rb") as f:
    for i, rec in enumerate(ijson.items(f, "item")):
        try:
            d = rec["publish_date"]
            y = int(d[:4])
            items = rec["items"]
            set_id.append(int(rec["set_id"]))
            user_id.append(int(rec["user"]["user_id"]))
            like.append(int(rec["like_num"]))
            n_items.append(len(items))
            year.append(y)
            ymd.append(int(d[:4]) * 10000 + int(d[5:7]) * 100 + int(d[8:10]))
            for it in items:
                item_id.append(int(it["item_id"]))
                cat1.append(int(it["category_id1"]))
                cat2.append(int(it["category_id2"]))
                price.append(int(float(it["price"])) if it.get("price") not in (None, "") else -1)
        except Exception as e:  # keep going, count malformed records
            bad += 1
            if bad <= 5:
                print("malformed record", i, repr(e), file=sys.stderr, flush=True)
        if (i + 1) % 250000 == 0:
            print(f"{i+1:,} records, {time.time()-t0:.0f}s, file pos {f.tell()/size*100:.0f}%", flush=True)

n = len(set_id)
print(f"parsed {n:,} outfits, {len(item_id):,} item slots, malformed={bad}, {time.time()-t0:.0f}s", flush=True)
np.savez_compressed(
    DST,
    set_id=np.array(set_id, dtype=np.int64),
    user_id=np.array(user_id, dtype=np.int64),
    like=np.array(like, dtype=np.int32),
    n_items=np.array(n_items, dtype=np.int16),
    year=np.array(year, dtype=np.int16),
    ymd=np.array(ymd, dtype=np.int32),
    item_id=np.array(item_id, dtype=np.int64),
    cat1=np.array(cat1, dtype=np.int16),
    cat2=np.array(cat2, dtype=np.int32),
    price=np.array(price, dtype=np.int64),
)
print("saved", DST, os.path.getsize(DST) / 1e6, "MB", flush=True)
