#!/usr/bin/env bash
# Step 0: download only the minimal samples: item catalog (52MB), year labels (1MB), ONE feature tar (<=448MB).
set -u
cd "$(dirname "$0")/.."
mkdir -p data/catalog data/year_labels data/features_tar_sample
BASE=https://research.zozo.com/data_release/shift15m

curl -sS -m 600 -o data/catalog/item_catalog.txt -w "item_catalog: HTTP %{http_code} %{size_download}B in %{time_total}s (%{speed_download} B/s)\n" \
  "$BASE/item_category_prediction/item_catalog.txt"
curl -sS -m 120 -o data/year_labels/year_classification_labels.tar.gz -w "year_labels: HTTP %{http_code} %{size_download}B\n" \
  "$BASE/year_classification/year_classification_labels.tar.gz"
( cd data/year_labels && tar xzf year_classification_labels.tar.gz && ls -la . | head -20 )

# one feature tar: pick the median-size one so the estimate is representative
TAR=$(python3 - <<'EOF'
rows=[l.split() for l in open("data/probe/tar_sizes.txt") if l.strip()]
rows=[(int(a),u) for a,u in rows if a.isdigit() and "tar_files" not in u and int(a)>1_000_000]
rows.sort()
print(rows[len(rows)//2][1])
EOF
)
echo "sample tar: $TAR"
curl -sS -m 1800 -o data/features_tar_sample/sample.tar.gz -w "sample tar: HTTP %{http_code} %{size_download}B in %{time_total}s (%{speed_download} B/s)\n" "$TAR"
ls -la data/features_tar_sample
echo "$TAR" > data/features_tar_sample/source_url.txt
