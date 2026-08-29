"""Keep only the final pilot entries in results/runs.json (drop debugging-era leftovers).

The pilot appends in a known order: HA x1, then full/nolag/nohybrid x `seeds` each.
This script keeps the LAST k entries per mode (k=1 for HA, k=2 otherwise) and
backs up the raw file. Run after run_pilot.sh completes.
"""
import json, os, shutil

RES = os.path.join(os.path.dirname(__file__), "results")
p = os.path.join(RES, "runs.json")
shutil.copy(p, p + ".raw.bak")
runs = json.load(open(p))
KEEP = {"HA": 1}
out = {}
for mode, entries in runs.items():
    k = KEEP.get(mode, 2)
    out[mode] = entries[-k:]
    print(mode, "kept", len(out[mode]), "of", len(entries))
json.dump(out, open(p, "w"), indent=2)
print("cleaned ->", p)
