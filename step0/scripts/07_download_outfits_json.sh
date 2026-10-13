#!/usr/bin/env bash
# Step 0: download the outfit metadata json (2.68GB, metadata only, NO image features).
# Needed to analyse leakage / per-year composition. Stored under data/ (not tracked by any git repo).
set -u
cd "$(dirname "$0")/.."
mkdir -p data/outfits
curl -sS -m 1800 -C - -o data/outfits/iqon_outfits.json -w "iqon_outfits.json: HTTP %{http_code} %{size_download}B in %{time_total}s (%{speed_download} B/s)\n" \
  https://research.zozo.com/data_release/shift15m/label/iqon_outfits.json
ls -la data/outfits
