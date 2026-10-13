"""Streaming SHIFT15M VGG16 feature extractor (resumable, verified, delete-after-verify).

  --local-tar PATH          dry run on an already-downloaded tar (NO network download; only a HEAD for the ETag)
  --run --authorized        full streaming of all 200 tars (refuses to start without --authorized)
  --finalize                merge shards -> one fp16 matrix + id index, check required-item coverage

Per tar: curl (resumable) -> verify size + S3 multipart ETag -> read whole gzip stream (CRC) -> keep needed items as fp16 ->
         write shard + stats -> reload shard & verify -> delete tar -> append to done.jsonl.
Licence: SHIFT15M data is CC BY-NC 4.0. Extracted features are derived data: keep local, do not publish or commit.
"""
import argparse, gzip, hashlib, json, os, subprocess, sys, tarfile, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE = "https://research.zozo.com/data_release/shift15m/vgg16-features"
OUT = os.path.join(ROOT, "data", "features_v1")
DIM = 4096
PART = 16 * 1024 * 1024


def head(url):
    h = subprocess.run(["curl", "-sSIL", "-m", "30", url], capture_output=True, text=True).stdout.replace("\r", "")
    d = {}
    for l in h.splitlines():
        if ":" in l:
            k, v = l.split(":", 1); d[k.strip().lower()] = v.strip()
    return int(d["content-length"]), d.get("etag", "").strip('"')


def multipart_etag(path, nparts):
    ds = []
    with open(path, "rb") as f:
        while True:
            b = f.read(PART)
            if not b:
                break
            ds.append(hashlib.md5(b).digest())
    if nparts == 1 and len(ds) == 1:
        return ds[0].hex()
    return hashlib.md5(b"".join(ds)).hexdigest() + f"-{len(ds)}"


def process_tar(name, required, keep_all, local_path=None):
    stem = name[:-len(".tar.gz")]
    os.makedirs(os.path.join(OUT, "shards"), exist_ok=True); os.makedirs(os.path.join(OUT, "_tmp"), exist_ok=True)
    stats = dict(tar=name, ok=False)
    t0 = time.time()
    size, etag = head(f"{BASE}/{name}")
    path = local_path or os.path.join(OUT, "_tmp", name)
    if local_path is None:
        r = subprocess.run(["curl", "-sS", "--fail", "-C", "-", "--retry", "5", "--retry-delay", "5", "-o", path, f"{BASE}/{name}"])
        if r.returncode != 0:
            stats["error"] = f"curl rc={r.returncode}"; return stats
    stats["size_ok"] = os.path.getsize(path) == size
    stats["etag_ok"] = multipart_etag(path, 1 if "-" not in etag else int(etag.split("-")[1])) == etag
    if not (stats["size_ok"] and stats["etag_ok"]):
        stats["error"] = "size/etag mismatch"; return stats
    ids, feats, n_in_tar, bad_dim, nonfinite, overflow, dirs = [], [], 0, 0, 0, 0, set()
    with tarfile.open(path, "r:gz") as tf:
        for m in tf:   # full sequential read => gzip CRC verified for the whole stream
            if not m.isfile():
                continue
            n_in_tar += 1
            dirs.add(os.path.dirname(m.name))
            iid = int(os.path.basename(m.name).split(".")[0])
            if not keep_all and iid not in required:
                continue
            v = np.asarray(json.loads(gzip.decompress(tf.extractfile(m).read())), dtype=np.float32)
            if v.shape != (DIM,):
                bad_dim += 1; continue
            if not np.isfinite(v).all():
                nonfinite += 1; continue
            if np.abs(v).max() > 65000:
                overflow += 1; continue
            ids.append(iid); feats.append(v.astype(np.float16))
    ids = np.asarray(ids, dtype=np.int64); F = np.stack(feats) if feats else np.zeros((0, DIM), np.float16)
    sp = os.path.join(OUT, "shards", stem + ".npz")
    np.savez(sp, ids=ids, feats=F)
    chk = np.load(sp)
    assert len(chk["ids"]) == len(ids) and chk["feats"].shape == F.shape and np.array_equal(chk["feats"][:5], F[:5]), "shard reload mismatch"
    stats.update(ok=True, n_in_tar=n_in_tar, n_kept=len(ids), bad_dim=bad_dim, nonfinite=nonfinite, overflow=overflow,
                 dirs=sorted(dirs), dir_matches_tar_name=(dirs == {stem}), seconds=round(time.time() - t0, 1),
                 shard_bytes=os.path.getsize(sp), required_found=int(sum(1 for i in ids if i in required)) if not keep_all else None)
    if local_path is None:
        os.remove(path)   # delete only after shard verified
    return stats


def load_required(path):
    return set(int(x) for x in open(path).read().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--required", default=os.path.join(ROOT, "manifests", "required_items_v1.txt"))
    ap.add_argument("--all-items", action="store_true", help="keep every item in each tar (~19 GB fp16) instead of only required ones")
    ap.add_argument("--local-tar"); ap.add_argument("--run", action="store_true"); ap.add_argument("--authorized", action="store_true")
    ap.add_argument("--finalize", action="store_true"); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    required = load_required(a.required)
    os.makedirs(OUT, exist_ok=True)
    if a.local_tar:
        name = os.path.basename(open(os.path.join(ROOT, "data/features_tar_sample/source_url.txt")).read().strip())
        s = process_tar(name, required, a.all_items, local_path=a.local_tar)
        print(json.dumps(s, indent=1)); return
    if a.run:
        if not a.authorized:
            sys.exit("Refusing to download 89.5 GB without --authorized (explicit user authorization required).")
        names = [l.strip() for l in open(os.path.join(ROOT, "data/probe/tar_files.txt")) if l.strip()]
        donef = os.path.join(OUT, "done.jsonl")
        done = {json.loads(l)["tar"] for l in open(donef)} if os.path.exists(donef) else set()
        todo = [n for n in names if n not in done]
        print(f"{len(done)} done, {len(todo)} to do, workers={a.workers}", flush=True)
        fails = []
        with ProcessPoolExecutor(a.workers) as ex:
            futs = {ex.submit(process_tar, n, required, a.all_items): n for n in todo}
            for f in as_completed(futs):
                s = f.result()
                if s.get("ok"):
                    with open(donef, "a") as fh:
                        fh.write(json.dumps(s) + "\n")
                else:
                    fails.append(s)
                print(("OK  " if s.get("ok") else "FAIL"), s["tar"], {k: s.get(k) for k in ("n_in_tar", "n_kept", "seconds", "error")}, flush=True)
        print("failures:", len(fails)); [print(x) for x in fails]
        sys.exit(1 if fails else 0)
    if a.finalize:
        shards = sorted(os.listdir(os.path.join(OUT, "shards")))
        shards = [s for s in shards if s.endswith(".npz")]
        # memory-safe: ids first, then stream shards into a sorted on-disk memmap
        ids_list = [np.load(os.path.join(OUT, "shards", s))["ids"] for s in shards]
        ids = np.concatenate(ids_list)
        assert len(np.unique(ids)) == len(ids), "an item_id appears in two tars"
        order = np.argsort(ids); sorted_ids = ids[order]
        rank = np.empty(len(ids), dtype=np.int64); rank[order] = np.arange(len(ids))   # global position of each concatenated row
        M = np.lib.format.open_memmap(os.path.join(OUT, "features_fp16.npy"), mode="w+", dtype=np.float16, shape=(len(ids), DIM))
        pos, finite = 0, True
        for s, il in zip(shards, ids_list):
            Fs = np.load(os.path.join(OUT, "shards", s))["feats"]
            assert len(Fs) == len(il)
            finite &= bool(np.isfinite(Fs.astype(np.float32)).all())
            M[rank[pos:pos + len(il)]] = Fs; pos += len(il)
        M.flush()
        np.save(os.path.join(OUT, "item_ids.npy"), sorted_ids)
        missing = sorted(required - set(sorted_ids.tolist()))
        open(os.path.join(OUT, "missing_required_items.txt"), "w").write("\n".join(map(str, missing)))
        print(f"shards={len(shards)} items={len(ids):,} required={len(required):,} missing_required={len(missing):,} "
              f"(rate {len(missing)/len(required)*100:.3f}%) | finite={finite}")


if __name__ == "__main__":
    main()
