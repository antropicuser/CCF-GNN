"""Shared data utilities: windowed METR-LA parquet -> tensors + masked metrics + graph loading."""
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA = os.path.join(os.path.dirname(__file__), "data")
RESULTS = os.path.join(os.path.dirname(__file__), "results")

IN_LAGS = [f"x_t-{k}_d0" for k in range(11, 0, -1)]          # 11 history steps
OUT_STEPS = [f"y_t+{k}_d0" for k in range(1, 13)]            # 12 horizons

def load_split(split="train", stride=1, max_windows=None):
    """Fast mmap'd loader from prepare_npz.py output. Sub-samples windows by `stride`.
    Returns X (W,N,11) f32, Y (W,N,12) f32, tod (W,) f32."""
    z = np.load(os.path.join(DATA, f"{split}_win.npz"), mmap_mode="r")
    X, Y, tod = z["X"], z["Y"], z["tod"]
    if stride > 1:
        X, Y, tod = X[::stride], Y[::stride], tod[::stride]
    if max_windows:
        X, Y, tod = X[:max_windows], Y[:max_windows], tod[:max_windows]
    return np.ascontiguousarray(X), np.ascontiguousarray(Y), np.ascontiguousarray(tod)

def zscore_fit(X):
    mu = X[X > 0].mean(); sd = X[X > 0].std()
    return float(mu), float(sd)

def masked_metrics(yhat, y):
    """DCRNN-standard: zeros in y are missing and excluded. yhat,y: (B,N,H) numpy f32."""
    m = (y > 0).astype(np.float32)
    absd = np.abs(yhat - y) * m
    mae = absd.sum() / max(m.sum(), 1)
    rmse = np.sqrt(((yhat - y) ** 2 * m).sum() / max(m.sum(), 1))
    mape = (absd / np.maximum(np.abs(y) * m, 1e-5)).sum() / max(m.sum(), 1) * 100
    return dict(MAE=float(mae), RMSE=float(rmse), MAPE=float(mape))

def load_graph(k_cap=10):
    """Load REAL CCF graph produced by ccf_graph.py; cap in-degree at k_cap by |rho*|."""
    g = np.load(os.path.join(RESULTS, "ccf_graph.npz"))
    mask, rho, tau = g["mask"], g["rho_agg"], g["tau_agg"]
    validated, confounded = g["validated"], g["confounded"]
    A_ccf = np.zeros_like(rho)
    A_ccf[mask] = rho[mask]
    N = mask.shape[0]
    keep = np.zeros_like(mask)
    for i in range(N):                       # cap incoming edges per target i
        js = np.where(mask[i])[0]
        if len(js) > k_cap:
            js = js[np.argsort(-rho[i, js])[:k_cap]]
        keep[i, js] = True
    mask_cap = keep
    A_ccf = A_ccf * mask_cap
    tau_eff = np.where(mask_cap, np.maximum(tau, 1), 1).astype(np.int64)
    mtype = np.zeros_like(mask_cap, dtype=np.float64)     # 1 validated, 0.5 confounded (for alpha init)
    mtype[validated & mask_cap] = 1.0
    mtype[confounded & mask_cap] = 0.5
    return dict(A_ccf=A_ccf.astype(np.float32), mask=mask_cap, tau=tau_eff,
                mtype=mtype.astype(np.float32), window_frac=None)
