#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PYTHONPATH=./pylibs python3 -u scripts/13_user_confound_check.py > results/step0_user_confound.txt 2> results/step0_user_confound.err
echo "exit=$?"
wc -l results/step0_user_confound.txt results/step0_user_confound.err
