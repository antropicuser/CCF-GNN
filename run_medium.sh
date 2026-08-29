#!/bin/bash
cd "$(dirname "$0")"
set -x
python3 train.py --mode ha    --stride 4 --epochs 1  --seeds 1 >> results/medium.log 2>&1
python3 train.py --mode full  --stride 4 --epochs 14 --seeds 1 --seed0 0 >> results/medium.log 2>&1
python3 train.py --mode gwnet --stride 4 --epochs 14 --seeds 1 --seed0 0 >> results/medium.log 2>&1
echo "SEED1 BLOCK DONE" >> results/medium.log
python3 train.py --mode full  --stride 4 --epochs 14 --seeds 1 --seed0 1 >> results/medium.log 2>&1
python3 train.py --mode gwnet --stride 4 --epochs 14 --seeds 1 --seed0 1 >> results/medium.log 2>&1
echo "SEED2 BLOCK DONE" >> results/medium.log
echo "MEDIUM DONE" >> results/medium.log
