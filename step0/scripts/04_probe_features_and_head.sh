#!/usr/bin/env bash
# Step 0: probe the VGG16 feature file list (HEAD only) and peek at the first bytes of the outfit json (Range request).
set -u
cd "$(dirname "$0")/.."
mkdir -p data/probe
BASE=https://research.zozo.com/data_release/shift15m

echo "===== feature filelist ====="
curl -sSL -m 60 -o data/probe/filelist.txt -w "filelist HTTP %{http_code} %{size_download}B\n" "$BASE/vgg16-features/filelist.txt"
wc -l data/probe/filelist.txt
head -3 data/probe/filelist.txt

echo "===== HEAD sizes of every tar (parallel x8) ====="
cat data/probe/filelist.txt | xargs -P 8 -I{} sh -c 'printf "%s %s\n" "$(curl -sSIL -m 60 "{}" | tr -d "\r" | grep -i -E "^content-length" | tail -1 | cut -d" " -f2)" "{}"' > data/probe/tar_sizes.txt
python3 - <<'EOF'
rows=[l.split() for l in open("data/probe/tar_sizes.txt") if l.strip()]
ok=[(int(a),u) for a,u in (r for r in rows if len(r)==2 and r[0].isdigit())]
print("tars with size:", len(ok), "of", len(rows))
tot=sum(a for a,_ in ok)
print("total GB:", round(tot/1e9,2), "min MB:", round(min(a for a,_ in ok)/1e6,1), "max MB:", round(max(a for a,_ in ok)/1e6,1))
EOF

echo "===== Range support + first bytes of iqon_outfits.json ====="
curl -sSIL -m 30 "$BASE/label/iqon_outfits.json" | tr -d '\r' | grep -i -E '^(accept-ranges|content-length)'
curl -sS -m 120 -r 0-4000000 -o data/probe/outfits_head.bin -w "range 4MB: HTTP %{http_code} %{size_download}B %{speed_download} B/s\n" "$BASE/label/iqon_outfits.json"
head -c 1500 data/probe/outfits_head.bin; echo
