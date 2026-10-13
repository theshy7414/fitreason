#!/bin/bash
cd ~/code/outfitmatch-research/step0
PYTHONPATH=./pylibs:./scripts python3 -u scripts/63_linear_baseline.py 2>&1 | grep --line-buffered -v -i -e "warn" -e "tensor.detach" -e "float(loss)" | tee results/exp2_linear_baseline_report.txt
