#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PYTHONPATH=./pylibs python3 -u scripts/11_measure_feature_extraction.py 2>&1 | tee results/step0_feature_timing.txt
echo "--- env ---"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
nproc
