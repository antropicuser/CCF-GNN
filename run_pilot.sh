#!/bin/bash
# Pilot study driver: reduced-scale runs (CPU) for the four paper configurations.
cd "$(dirname "$0")"
set -x
python3 train.py --mode ha        --stride 30 --epochs 2 --seeds 1 >> results/pilot.log 2>&1
for mode in full nolag nohybrid; do
  python3 train.py --mode $mode --stride 30 --epochs 6 --seeds 2 >> results/pilot.log 2>&1
done
echo "PILOT DONE" >> results/pilot.log
