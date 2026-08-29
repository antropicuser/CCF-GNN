"""Download METR-LA (standard split) from the Hugging Face mirror of the DCRNN release."""
import os, sys, urllib.request

BASE = "https://huggingface.co/datasets/witgaw/METR-LA/resolve/main"
FILES = ["train.parquet", "val.parquet", "test.parquet",
         "sensor_graph/adj_mx.npy", "sensor_graph/distances.csv", "sensor_graph/sensor_locations.csv"]

def main(dest="data"):
    os.makedirs(dest, exist_ok=True); os.makedirs(os.path.join(dest, "sensor_graph"), exist_ok=True)
    for f in FILES:
        out = os.path.join(dest, f)
        if os.path.exists(out): print("skip", f); continue
        print("downloading", f, flush=True)
        urllib.request.urlretrieve(f"{BASE}/{f}", out)
    print("done ->", dest)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
