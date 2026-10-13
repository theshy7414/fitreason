#!/usr/bin/env bash
# Step 0: read the official download scripts, then probe each URL with HEAD only (no data download).
set -u
cd "$(dirname "$0")/.."
BASE=https://raw.githubusercontent.com/st-tech/zozo-shift15m/main/scripts
mkdir -p docs/official/scripts
for f in README.md download_cnn_features.sh download_shift15m_outfits.sh download_shift15m_outfits_with_tags.sh \
         download_year_classification_label.sh download_item_category_prediction_catalog.sh \
         download_num_likes_regression.sh download_sum_prices_regression.sh; do
  curl -sSL -m 30 -o "docs/official/scripts/$f" "$BASE/$f"
done
echo "===== script contents ====="
for f in docs/official/scripts/*.sh; do echo "--- $f"; cat "$f"; echo; done
echo "===== scripts/README.md ====="
cat docs/official/scripts/README.md
echo "===== URL probe (HEAD) ====="
grep -ho -E 'https?://[^ "'"'"')]+' docs/official/scripts/*.sh | sort -u | while read -r u; do
  curl -sSIL -m 30 -o /dev/null -w "%{http_code} %{size_download} content-length-header-below  $u\n" "$u"
  curl -sSIL -m 30 "$u" | grep -i -E '^(HTTP|content-length|content-type|last-modified)' | tr -d '\r' | sed 's/^/    /'
done
