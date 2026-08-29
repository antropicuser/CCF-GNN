#!/bin/bash
# Resumable driver: finishes medium comparison + runs the windowed (6-map) variant.
cd "$(dirname "$0")"
NGW=$(python3 -c "import json;print(len(json.load(open('results/runs.json')).get('gwnet',[])))")
if [ "$NGW" -lt 2 ]; then
  python3 train.py --mode gwnet --stride 4 --epochs 14 --seeds 1 --seed0 1 >> results/medium.log 2>&1
fi
NW=$(python3 -c "import json;print(len(json.load(open('results/runs.json')).get('windowed',[])))")
if [ "$NW" -lt 2 ]; then
  python3 train.py --mode windowed --stride 4 --epochs 14 --seeds $((2-NW)) --seed0 $NW >> results/phase1.log 2>&1
fi
# also re-run the lost CCF-GNN seed2 (full)
NF=$(python3 -c "import json;print(len(json.load(open('results/runs.json')).get('full',[])))")
if [ "$NF" -lt 2 ]; then
  python3 train.py --mode full --stride 4 --epochs 14 --seeds $((2-NF)) --seed0 $NF >> results/medium.log 2>&1
fi
python3 make_medium_table.py >> results/phase1.log 2>&1
echo "PHASE1 DONE" >> results/phase1.log
