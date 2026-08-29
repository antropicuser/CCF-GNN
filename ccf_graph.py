"""
ccf_graph.py -- Windowed cross-correlation (CCF) graph construction on REAL METR-LA data.

Implements the CCF-GNN graph-construction stage exactly as specified in the paper:
  1. Six diurnal windows (regimes) on time-of-day.
  2. Per-window z-standardisation (missing values = 0 in raw METR-LA are masked as NaN).
  3. FFT-free direct cross-correlation at lags tau = 1..L via shifted matrix products.
  4. Optimal lag tau*(w) and peak correlation rho*(w) per directed pair per window.
  5. Two-tailed t-test on rho*, Benjamini-Hochberg FDR correction across all pairs.
  6. Edge criterion: |rho*| > rho_thresh AND q < 0.05 (FDR-corrected).
  7. Conditional-independence screening on road-network intermediates (partial corr).
  8. Window-aggregated adjacency (n_w * |rho*| weighting) + per-window lag tables.
  9. Diagnostics: lag distribution, sparsity, road-corridor agreement, ADF failure rates.

Outputs: results/graph_stats.json, results/ccf_graph.npz, figures/*.pdf,
         ../../paper/tables/graph_stats.tex
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

RNG = np.random.default_rng(0)
DATA = os.path.join(os.path.dirname(__file__), "data")
OUT = os.path.join(os.path.dirname(__file__), "results")
FIG = os.path.join(os.path.dirname(__file__), "figures")
TAB = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "paper", "tables"))
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

# ----------------------------- config ---------------------------------------
WINDOWS = [("w1", 0, 6), ("w2", 6, 8), ("w3", 8, 10), ("w4", 10, 15), ("w5", 15, 19), ("w6", 19, 24)]
L_MAX = 12          # max lag (12 x 5 min = 60 min)
RHO_THRESH = 0.30   # effect-size filter
FDR_Q = 0.05        # BH-FDR level
CI_DROP_RATIO = 0.5 # partial corr drop ratio -> flag confounded
CI_MAX_COND = 3     # max conditioning nodes (road-path intermediates)
MIN_OBS = 500       # min valid observations per window per sensor

def load_series():
    """Reconstruct continuous per-sensor series from windowed train parquet."""
    cols = ["node_id", "t0_timestamp", "x_t-1_d0"]
    df = pd.read_parquet(os.path.join(DATA, "train.parquet"), columns=cols)
    df["t0_timestamp"] = pd.to_datetime(df["t0_timestamp"])
    piv = df.pivot_table(index="t0_timestamp", columns="node_id", values="x_t-1_d0", aggfunc="first")
    piv = piv.sort_index()
    Z = piv.to_numpy(dtype=np.float64)          # (T, N)
    Z[Z == 0.0] = np.nan                        # raw METR-LA encodes missing as 0
    hours = piv.index.hour.to_numpy()
    print(f"series: T={Z.shape[0]}  N={Z.shape[1]}  missing={np.isnan(Z).mean()*100:.2f}%")
    return Z, hours

def window_mask(hours, lo, hi):
    return (hours >= lo) & (hours < hi)

def cross_corr_lags(Xw, lags):
    """Xw: (T_w, N) standardised (NaN->0, validity mask returned).
    Returns corr[lags, N, N] where corr[k, i, j] = corr(x_i(t), x_j(t-lag_k))."""
    Tw, N = Xw.shape
    out = np.zeros((len(lags), N, N), dtype=np.float64)
    for k, lag in enumerate(lags):
        if lag >= Tw: continue
        A = Xw[lag:]          # x(t)
        B = Xw[:-lag]         # x(t-lag)
        C = (A.T @ B) / Tw    # (N, N)
        out[k] = C
    return out

def bh_fdr(pvals):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=np.float64)
    out[order] = np.minimum(q, 1.0)
    return out

def main():
    Z, hours = load_series()
    T, N = Z.shape
    lags = list(range(1, L_MAX + 1))
    dt_minutes = 5

    # road graph for path-based conditioning & corridor agreement
    A = np.asarray(np.load(os.path.join(DATA, "adj_mx.npy"), allow_pickle=True), dtype=np.float64)
    Nchk = A.shape[0]
    assert Nchk == N, f"adjacency {A.shape} != series nodes {N}"
    D, pred = shortest_path(csr_matrix(A), method="D", unweighted=True, return_predecessors=True)

    def path_nodes(j, i, cap=CI_MAX_COND):
        """intermediate nodes on shortest path j -> i"""
        if not np.isfinite(D[j, i]) or D[j, i] < 2:
            return []
        nodes, cur = [], i
        while cur != j and len(nodes) < 64:
            cur = pred[j, cur]
            if cur < 0: return []
            if cur != j: nodes.append(int(cur))
        return nodes[:cap]

    per_window = {}
    lag_hist_all = np.zeros(L_MAX, dtype=np.int64)
    edges_union = set()
    A_acc = np.zeros((N, N)); W_acc = np.zeros((N, N))
    adf_fail_before, adf_fail_after, adf_tested = 0, 0, 0
    ts = pd.Timestamp("2012-03-01")  # placeholder; hour only matters

    from statsmodels.tsa.stattools import adfuller
    for wname, lo, hi in WINDOWS:
        m = window_mask(hours, lo, hi)
        Xw = Z[m]
        valid = ~np.isnan(Xw)
        Xf = np.where(valid, Xw, 0.0)
        mu = Xf.sum(0) / np.maximum(valid.sum(0), 1)
        var = ((Xf - mu) ** 2 * valid).sum(0) / np.maximum(valid.sum(0), 1)
        sd = np.sqrt(np.maximum(var, 1e-8))
        Xs = (Xf - mu) / sd
        Xs = np.where(valid, Xs, 0.0)
        Tw = Xs.shape[0]

        # ADF test (sample up to 60 sensors for speed)
        idx = RNG.choice(N, size=min(60, N), replace=False)
        for i in idx:
            s = Xw[:, i]; s = s[~np.isnan(s)]
            if len(s) < MIN_OBS: continue
            adf_tested += 1
            try:
                if adfuller(s, autolag="AIC")[1] > 0.05: adf_fail_before += 1
                d1 = np.diff(s)
                if adfuller(d1, autolag="AIC")[1] > 0.05: adf_fail_after += 1
            except Exception:
                pass

        # CCF at lags 1..L  (excludes tau=0 to avoid instantaneous confounds)
        C = cross_corr_lags(Xs, lags)                       # (L, N, N)
        absC = np.abs(C)
        k_best = absC.argmax(axis=0)                        # (N, N) arg over lags
        rho = np.take_along_axis(C, k_best[None], axis=0)[0]  # signed peak
        tau = np.take_along_axis(np.array(lags, dtype=float)[:, None, None].repeat(N, 1).repeat(N, 2),
                                 k_best[None], axis=0)[0]

        # significance: t-test on Pearson rho with T_w - 2 dof, then BH-FDR
        dof = max(Tw - 2, 3)
        with np.errstate(divide="ignore", invalid="ignore"):
            tstat = rho * np.sqrt(dof / np.maximum(1 - rho**2, 1e-12))
        pval = 2 * sps.t.sf(np.abs(tstat), df=dof)
        qval = bh_fdr(pval.ravel()).reshape(N, N)

        n0 = valid.sum(0)
        keep = (n0 >= MIN_OBS)[:, None] & (n0 >= MIN_OBS)[None, :] \
               & (np.abs(rho) > RHO_THRESH) & (qval < FDR_Q) & (tau > 0)
        np.fill_diagonal(keep, False)

        ei, ej = np.where(keep)   # edge j -> i : corr(x_i(t), x_j(t-tau))
        per_window[wname] = dict(
            n_obs=int(Tw),
            candidate_pairs=int(N * (N - 1)),
            n_significant_raw=int((pval < 0.05).sum() - N),  # excl. diagonal
            n_edges=int(keep.sum()),
            sparsity_pct=float(100 * (1 - keep.sum() / (N * (N - 1)))),
            avg_degree=float(keep.sum() / N),
        )
        for i, j in zip(ei, ej):
            edges_union.add((int(i), int(j)))
            wgt = Tw * abs(rho[i, j])
            A_acc[i, j] += wgt * abs(rho[i, j]) / max(abs(rho[i, j]), 1e-9)  # |rho*|-weighted
            W_acc[i, j] += wgt
        lag_hist_all += np.bincount(tau[keep].astype(int), minlength=L_MAX + 1)[1:]
        per_window[wname]["mean_tau_min"] = float(5 * tau[keep].mean()) if keep.sum() else None
        print(wname, per_window[wname])

    # ---------------- aggregated graph + lag table ----------------
    E = len(edges_union)
    tau_agg = np.zeros((N, N)); rho_agg = np.zeros((N, N)); present = np.zeros((N, N), bool)
    # recompute quickly to fill aggregated tau/rho (second pass, cheap)
    for wname, lo, hi in WINDOWS:
        m = window_mask(hours, lo, hi)
        Xw = Z[m]; valid = ~np.isnan(Xw); Xf = np.where(valid, Xw, 0.0)
        mu = Xf.sum(0) / np.maximum(valid.sum(0), 1)
        sd = np.sqrt(np.maximum(((Xf - mu) ** 2 * valid).sum(0) / np.maximum(valid.sum(0), 1), 1e-8))
        Xs = np.where(valid, (Xf - mu) / sd, 0.0)
        C = cross_corr_lags(Xs, lags)
        k_best = np.abs(C).argmax(axis=0)
        rho = np.take_along_axis(C, k_best[None], axis=0)[0]
        tau = k_best + 1
        for (i, j) in edges_union:
            tau_agg[i, j] += Xs.shape[0] * tau[i, j]
            rho_agg[i, j] += Xs.shape[0] * abs(rho[i, j])
            present[i, j] = True
    Tw_tot = sum(dict((w, per_window[w]["n_obs"]) for w in per_window).values())
    for (i, j) in edges_union:
        tau_agg[i, j] /= Tw_tot; rho_agg[i, j] /= Tw_tot

    # ---------------- conditional independence screening ----------------
    # condition on up to CI_MAX_COND road-path intermediates of j->i
    validated = np.zeros((N, N), bool); confounded = np.zeros((N, N), bool)
    # reuse standardised full-day series for partial correlations
    valid_all = ~np.isnan(Z); Zf = np.where(valid_all, Z, 0.0)
    mu = Zf.sum(0) / np.maximum(valid_all.sum(0), 1)
    sd = np.sqrt(np.maximum(((Zf - mu) ** 2 * valid_all).sum(0) / np.maximum(valid_all.sum(0), 1), 1e-8))
    S = np.where(valid_all, (Zf - mu) / sd, 0.0)
    for (i, j) in sorted(edges_union):
        tlag = max(int(round(tau_agg[i, j])), 1)
        x_i = S[tlag:, i]           # x_i(t)
        x_j = S[:-tlag, j]          # x_j(t - tlag)
        inter = path_nodes(j, i)
        if len(inter) == 0 or len(x_i) < MIN_OBS:
            validated[i, j] = True; continue
        M = np.column_stack([S[:-tlag, k] for k in inter])   # intermediates at same lag
        Xd = np.column_stack([np.ones(len(x_i)), M])
        beta_i, *_ = np.linalg.lstsq(Xd, x_i, rcond=None)
        beta_j, *_ = np.linalg.lstsq(Xd, x_j, rcond=None)
        r_i = x_i - Xd @ beta_i; r_j = x_j - Xd @ beta_j
        pr = float(np.corrcoef(r_i, r_j)[0, 1])
        if abs(pr) < CI_DROP_RATIO * abs(rho_agg[i, j]): confounded[i, j] = True
        else: validated[i, j] = True

    # ---------------- corridor agreement (road distance vs lag) ----------------
    loc = pd.read_csv(os.path.join(DATA, "sensor_locations.csv"))
    locxy = np.radians(loc.iloc[:, 1:3].to_numpy(dtype=float))
    checked = feas = 0
    for (i, j) in sorted(edges_union):
        checked += 1
        if i < len(locxy) and j < len(locxy):
            dlat = locxy[i, 0] - locxy[j, 0]
            dlon = locxy[i, 1] - locxy[j, 1]
            a = np.sin(dlat/2)**2 + np.cos(locxy[i,0])*np.cos(locxy[j,0])*np.sin(dlon/2)**2
            d_km = 6371.0 * 2 * np.arcsin(np.sqrt(a))
            v = 15.0  # km/h conservative shockwave speed
            if d_km <= v * (tau_agg[i, j] * dt_minutes / 60.0) + 1.5:
                feas += 1
    eu = np.array(sorted(edges_union))
    lag_vals = tau_agg[eu[:, 0], eu[:, 1]] * dt_minutes

    # ---------------- save everything ----------------
    pct = np.percentile(lag_vals, [50, 75, 95]) if len(lag_vals) else [0, 0, 0]
    modal = int(np.bincount((lag_vals / dt_minutes).astype(int)).argmax()) if len(lag_vals) else 0
    stats = dict(
        dataset="METR-LA (train split, witgaw/HF mirror of DCRNN release)",
        n_sensors=int(N), n_timesteps=int(T),
        missing_pct=float(np.isnan(Z).mean() * 100),
        windows=per_window,
        aggregated=dict(n_edges=int(E), sparsity_pct=float(100 * (1 - E / (N * (N - 1)))),
                        avg_degree=float(E / N),
                        validated=int(validated.sum()), confounded=int(confounded.sum()),
                        median_lag_min=float(pct[0]), p75_lag_min=float(pct[1]), p95_lag_min=float(pct[2]),
                        modal_lag_min=int(modal * dt_minutes),
                        lag_feasibility_pct=float(100 * feas / max(checked, 1))),
        adf=dict(tested=adf_tested, fail_before=int(adf_fail_before), fail_after=int(adf_fail_after),
                 fail_before_pct=float(100 * adf_fail_before / max(adf_tested, 1)),
                 fail_after_pct=float(100 * adf_fail_after / max(adf_tested, 1))),
    )
    with open(os.path.join(OUT, "graph_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    np.savez_compressed(os.path.join(OUT, "ccf_graph.npz"),
                        A_ccf=A_acc / np.maximum(W_acc, 1e-9), mask=present,
                        validated=validated, confounded=confounded,
                        tau_agg=tau_agg, rho_agg=rho_agg)
    print(json.dumps(stats["aggregated"], indent=2))

    # ---------------- LaTeX table fragment (real numbers) ----------------
    lines = [
        "% auto-generated by experiments/ccf_graph.py -- REAL METR-LA numbers",
        "\\begin{table}[t]",
        "\\centering\\small",
        "\\caption{Window-wise CCF graph statistics computed on the METR-LA training split "
        "($L{=}12$, $|\\rho^{*}|{>}0.3$, BH--FDR $q{<}0.05$, $\\tau{\\ge}1$). Reproducible with "
        "\\texttt{experiments/ccf\\_graph.py}.}",
        "\\label{tab:graph_stats}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Window & Hours & Obs. & Edges & Avg. deg. & Sparsity & Mean $\\tau^{*}$ (min) \\\\",
        "\\midrule",
    ]
    for wname, lo, hi in WINDOWS:
        d = per_window[wname]
        mt = d["mean_tau_min"]
        lines.append(
            f"{wname} & {lo:02d}:00--{hi:02d}:00 & {d['n_obs']:,} & {d['n_edges']:,} & "
            f"{d['avg_degree']:.2f} & {d['sparsity_pct']:.1f}\\% & "
            f"{('%.1f' % mt) if mt is not None else '--'} \\\\"
        )
    a = stats["aggregated"]
    lines += [
        "\\midrule",
        f"Aggregated & -- & {stats['n_timesteps']:,} & {a['n_edges']:,} & {a['avg_degree']:.2f} & "
        f"{a['sparsity_pct']:.1f}\\% & {a['median_lag_min']:.1f} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    with open(os.path.join(TAB, "graph_stats.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", os.path.join(TAB, "graph_stats.tex"))

if __name__ == "__main__":
    main()
