#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export PYTHONPATH=./pylibs
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv 2>&1 | tail -2
echo "cpus: $(nproc)"; free -g | sed -n 2p
python3 -c "import scipy;print('scipy',scipy.__version__)" 2>&1 | tail -1
python3 -c "import numpy;print('numpy',numpy.__version__)"
