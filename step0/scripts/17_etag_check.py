"""Preflight: (1) are the 'checksum' sibling URLs real files or a catch-all 200? (2) can S3 multipart ETag be used as integrity check?"""
import hashlib, os, subprocess, sys
ROOT = os.path.join(os.path.dirname(__file__), "..")
B = "https://research.zozo.com/data_release/shift15m/vgg16-features"
for n in ["md5sum.txt", "SHA256SUMS", "definitely_not_a_real_file_12345.txt"]:
    h = subprocess.run(["curl", "-sSIL", "-m", "15", f"{B}/{n}"], capture_output=True, text=True).stdout.replace("\r", "")
    keep = [l for l in h.splitlines() if l.lower().startswith(("http", "content-type", "content-length", "etag"))]
    print(n, "->", keep)

p = os.path.join(ROOT, "data/features_tar_sample/sample.tar.gz")
size = os.path.getsize(p)
ds = []
with open(p, "rb") as f:
    while True:
        b = f.read(16 * 1024 * 1024)
        if not b:
            break
        ds.append(hashlib.md5(b).digest())
print("file size:", size)
print("computed multipart ETag (16 MiB parts):", hashlib.md5(b"".join(ds)).hexdigest() + f"-{len(ds)}")
print("server ETag                           : 68d6ea0739bc5868419e813a822b7154-27")
