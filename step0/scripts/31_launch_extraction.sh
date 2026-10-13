#!/usr/bin/env bash
# Authorized by user (2026-10-12): stream all 200 tars, keep ALL items as fp16. Resumable: rerun this script to continue.
cd "$(dirname "$0")/.."
export PYTHONPATH=./pylibs
nohup python3 scripts/30_stream_extract_features.py --run --authorized --all-items --workers 4 >> results/exp1_extraction.log 2>&1 &
echo "started pid $!"
