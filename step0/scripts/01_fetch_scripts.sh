#!/usr/bin/env bash
# Step 0: inspect the official download mechanism WITHOUT downloading any dataset.
set -u
cd "$(dirname "$0")/.."
mkdir -p docs/official
BASE=https://raw.githubusercontent.com/st-tech/zozo-shift15m/main
for f in LICENSE.CC LICENSE.MIT scripts/download_all.sh DEVELOPMENT.md; do
  out="docs/official/$(echo "$f" | tr '/' '_')"
  curl -sSL -m 30 -o "$out" -w "$f HTTP %{http_code} %{size_download}B\n" "$BASE/$f"
done
echo "--- scripts dir (GitHub API) ---"
curl -sSL -m 30 https://api.github.com/repos/st-tech/zozo-shift15m/contents/scripts | python3 -c "import sys,json; [print(x['name'], x['type']) for x in json.load(sys.stdin)]" 2>&1 | head -40
echo "--- repo info ---"
curl -sSL -m 30 https://api.github.com/repos/st-tech/zozo-shift15m | python3 -c "import sys,json; d=json.load(sys.stdin); print({k:d.get(k) for k in ['pushed_at','updated_at','archived','license','default_branch','size']})" 2>&1
echo "--- download_all.sh ---"
cat docs/official/scripts_download_all.sh 2>/dev/null | head -60
