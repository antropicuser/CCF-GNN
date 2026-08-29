#!/bin/bash
cd "$(dirname "$0")"
python3 -u robustness_eval.py > results/robust.log 2>&1
echo "ROBUST DONE" >> results/robust.log
