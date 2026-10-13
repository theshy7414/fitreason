#!/usr/bin/env bash
# Preflight: (a) scikit-learn into local pylibs (needed to mimic the official train_test_split exactly)
#            (b) does the host publish any checksum? (full HEAD headers of one tar + tar_files.txt contents)
set -u
cd "$(dirname "$0")/.."
python3 -m pip install -q --target ./pylibs scikit-learn 2>&1 | grep -v -i "dependency resolver\|onnxruntime" | tail -2
PYTHONPATH=./pylibs python3 -c "import sklearn; print('sklearn', sklearn.__version__)"
BASE=https://research.zozo.com/data_release/shift15m/vgg16-features
echo "--- full HEAD headers of the sample tar ---"
curl -sSIL -m 30 "$(cat data/features_tar_sample/source_url.txt)" | tr -d '\r'
echo "--- tar_files.txt ---"
curl -sSL -m 30 "$BASE/tar_files.txt" -o data/probe/tar_files.txt -w "HTTP %{http_code} %{size_download}B\n"
wc -l data/probe/tar_files.txt; head -3 data/probe/tar_files.txt
echo "--- any checksum-looking siblings? (HEAD only) ---"
for n in md5sum.txt MD5SUMS sha256sum.txt SHA256SUMS checksums.txt checksum.txt md5.txt; do
  printf "%s -> " "$n"; curl -sSIL -m 15 -o /dev/null -w "%{http_code}\n" "$BASE/$n"
done
echo "--- do tar_files.txt names match the 200 tars in filelist.txt? ---"
python3 - <<'EOF'
import os
a = set(l.strip() for l in open("data/probe/tar_files.txt") if l.strip())
b = set(os.path.basename(l.strip()) for l in open("data/probe/filelist.txt") if l.strip() and "tar_files" not in l)
print("tar_files.txt entries:", len(a), "| filelist tars:", len(b), "| same set:", a == b, "| only-in-tar_files:", len(a - b), "| only-in-filelist:", len(b - a))
EOF
