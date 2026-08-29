#!/bin/bash
cd "$(dirname "$0")"
set -x
NGW=$(python3 -c "import json;print(len(json.load(open('results/runs.json')).get('gwnet',[])))")
if [ "$NGW" -lt 2 ]; then
  python3 train.py --mode gwnet --stride 4 --epochs 14 --seeds 1 --seed0 1 >> results/medium.log 2>&1
fi
NF=$(python3 -c "import json;print(len(json.load(open('results/runs.json')).get('full',[])))")
if [ "$NF" -lt 2 ]; then
  python3 train.py --mode full --stride 4 --epochs 14 --seeds 1 --seed0 1 >> results/medium.log 2>&1
fi
python3 robustness_eval.py >> results/phase2.log 2>&1
python3 make_medium_table.py >> results/phase2.log 2>&1
echo "PHASE2 DONE" >> results/phase2.log
