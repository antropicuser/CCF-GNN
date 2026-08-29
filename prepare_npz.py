"""One-time conversion of windowed METR-LA parquet splits -> compact npz (float32).

Run once before training:   python prepare_npz.py
Produces data/{split}_win.npz with arrays X (W,N,11) f32, Y (W,N,12) f32, tod (W,) f32.
Streams the parquet in row-group batches so peak RSS stays low.
"""
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA = os.path.join(os.path.dirname(__file__), "data")
IN_LAGS = [f"x_t-{k}_d0" for k in range(11, 0, -1)]
OUT_STEPS = [f"y_t+{k}_d0" for k in range(1, 13)]

def convert(split):
    pf = pq.ParquetFile(os.path.join(DATA, f"{split}.parquet"))
    cols = ["node_id", "t0_timestamp"] + IN_LAGS + OUT_STEPS
    # pass 1: ordered unique t0s
    t0_list = []
    seen = set()
    for batch in pf.iter_batches(batch_size=400_000, columns=["t0_timestamp"]):
        for t in batch.column(0).to_numpy(zero_copy_only=False):
            if t not in seen:
                seen.add(t); t0_list.append(t)
    t0_arr = np.array(sorted(t0_list)); del seen, t0_list
    W = len(t0_arr)
    keys = pd.to_datetime(t0_arr).asi8
    t0_idx = {t: k for k, t in enumerate(keys.tolist())}
    head = pf.schema_arrow
    N = 207
    X = np.zeros((W, N, len(IN_LAGS)), dtype=np.float32)
    Y = np.zeros((W, N, len(OUT_STEPS)), dtype=np.float32)
    seen_w = np.zeros(W, dtype=bool)
    for batch in pf.iter_batches(batch_size=400_000, columns=cols):
        d = {c: batch.column(c).to_numpy(zero_copy_only=False) for c in cols}
        ti = np.fromiter((t0_idx.get(t, -1) for t in pd.to_datetime(d["t0_timestamp"]).asi8),
                         dtype=np.int64, count=len(d["t0_timestamp"]))
        ok = ti >= 0
        if not ok.any(): continue
        ni, wi = d["node_id"][ok], ti[ok]
        for k, name in enumerate(IN_LAGS):  X[wi, ni, k] = d[name][ok]
        for k, name in enumerate(OUT_STEPS): Y[wi, ni, k] = d[name][ok]
        seen_w[wi] = True
    X, Y = X[seen_w], Y[seen_w]
    t0s = pd.to_datetime(t0_arr)[seen_w]
    hour = t0s.hour + t0s.minute / 60.0
    tod = (hour / 24.0).astype(np.float32)
    np.savez_compressed(os.path.join(DATA, f"{split}_win.npz"), X=X, Y=Y, tod=tod)
    print(split, "->", X.shape, Y.shape)

if __name__ == "__main__":
    for s in ["train", "val", "test"]:
        convert(s)
