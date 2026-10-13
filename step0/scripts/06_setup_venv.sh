#!/usr/bin/env bash
# Step 0: isolated Python deps for analysis scripts, installed into ./pylibs (no sudo, no system changes,
# nothing installed into the v1 repo). Use with: PYTHONPATH=./pylibs python3 script.py
set -eu
cd "$(dirname "$0")/.."
rm -rf .venv
python3 -m pip install -q --target ./pylibs numpy pandas ijson
PYTHONPATH=./pylibs python3 -c "import numpy, pandas, ijson; print('numpy', numpy.__version__, 'pandas', pandas.__version__, 'ijson', ijson.__version__)"
