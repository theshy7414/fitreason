#!/usr/bin/env bash
# Step 0: download ONLY the shift15m python package (no dataset) and read how Set2SetMatching gets its data.
set -u
cd "$(dirname "$0")/.."
mkdir -p pkg && cd pkg
[ -f shift15m-0.2.0-py3-none-any.whl ] || python3 -m pip download shift15m==0.2.0 --no-deps -d . -q 2>&1 | tail -2
python3 - <<'EOF'
import zipfile, glob
for w in glob.glob("shift15m-*.whl"):
    zipfile.ZipFile(w).extractall("unpacked")
EOF
echo "===== package files ====="
find unpacked -type f | sort
echo "===== METADATA ====="
grep -i -E '^(Requires-Dist|License|Requires-Python|Version)' unpacked/*.dist-info/METADATA
