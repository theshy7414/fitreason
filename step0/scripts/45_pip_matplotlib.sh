#!/usr/bin/env bash
cd "$(dirname "$0")/.."
python3 -m pip install -q --target ./pylibs --upgrade matplotlib 2>&1 | grep -v -i "dependency resolver\|onnxruntime\|WARNING" | tail -3
PYTHONPATH=./pylibs python3 -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; print('matplotlib', matplotlib.__version__, 'ok')"
