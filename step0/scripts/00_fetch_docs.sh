#!/usr/bin/env bash
# Step 0: fetch the official SHIFT15M documentation (small text files only).
set -u
cd "$(dirname "$0")/.."
BASE=https://raw.githubusercontent.com/st-tech/zozo-shift15m/main
for f in README.md DATASHEET.md LICENSE; do
  curl -sSL -m 30 -o "docs/$f" -w "$f HTTP %{http_code} %{size_download}B\n" "$BASE/$f"
done
ls -la docs
python3 -m venv --help >/dev/null 2>&1 && echo venv-ok || echo venv-missing
df -h ~ | tail -1
free -g | sed -n 2p
