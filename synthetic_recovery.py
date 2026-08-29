"""Synthetic ground-truth recovery study (the paper's core evaluation).

Generates traffic-like surrogate series on the REAL METR-LA road topology with a KNOWN
directed, time-lagged causal graph (shockwave-style propagation along road edges) plus
deliberate confounders (shared latent drivers creating long-range synchronous correlations).
Then runs the SAME CCF+screening pipeline used on real data and measures, against ground
truth, the precision/recall of causal discovery under successive screening stages, and the
accuracy of recovered propagation lags.

Stages compared:
  V1  p-value only (q<0.05, FDR-corrected)          -- no effect size
  V2  effect size only (|rho*|>0.3)                 -- no multiplicity control
  V3  full admission (|rho*|>0.3 AND q<0.05)        -- paper criterion
  V4  V3 + conditional-independence confound screen -- validated edges only
  V5  V4 + degree cap K=10                          -- the deployed graph

Outputs: results/synthetic_stats.json, ../../paper/tables/synthetic.tex,
         ../../paper/figures/synthetic_pr.pdf
"""
import json, os, sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

sys.path.insert(0, os.path.dirname(__file__))
from ccf_graph import WINDOWS, L_MAX, RHO_THRESH, FDR_Q, CI_DROP_RATIO, cross_corr_lags, bh_fdr

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
TAB = os.path.abspath(os.path.join(HERE, "..", "paper", "tables"))
FIG = os.path.abspath(os.path.join(HERE, "..", "paper", "figures"))
os.makedirs(RES, exist_ok=True); os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
import os as _os
SEED = int(_os.environ.get("SYN_SEED", "0"))

# ------------------------------- generator ----------------------------------
def generate(N=207, T=9600, rng=None):
    A = np.asarray(np.load(os.path.join(HERE, "data", "adj_mx.npy")), dtype=float)
    D, pred = shortest_path(csr_matrix(A), method="D", unweighted=True, return_predecessors=True)
    dt_min = 5

    true_edges = {}          # (j, i) -> (tau, beta)
    for i in range(N):
        nbrs = [j for j in range(N) if A[j, i] > 0 and j != i]
        if not nbrs: continue
        k = rng.integers(3, 6)
        parents = rng.choice(nbrs, size=min(k, len(nbrs)), replace=False)
        for j in parents:
            hops = max(D[j, i], 1)
            rate = rng.uniform(1.5, 2.5)                  # lag steps per hop
            tau = int(np.clip(round(hops * rate), 1, L_MAX))
            beta = rng.uniform(0.30, 0.55)
            true_edges[(int(j), int(i))] = (tau, beta)

    alpha = rng.uniform(0.15, 0.35, size=N)
    # stability: rescale in-weights so alpha + sum(beta_in) <= 0.85
    in_sum = np.zeros(N)
    for (j, i), (tau, beta) in true_edges.items(): in_sum[i] += beta
    scale = np.minimum(1.0, (0.85 - alpha) / np.maximum(in_sum, 1e-9))
    true_edges = {(j, i): (tau, beta * scale[i]) for (j, i), (tau, beta) in true_edges.items()}

    # confounders: C latent AR(1) drivers, each hitting a random ~25% cluster of nodes
    C = 5
    zphi = 0.9
    zsd = 0.5
    zw = np.zeros((N, C)); zd = np.zeros(C, dtype=int)
    for c in range(C):
        cluster = rng.choice(N, size=int(0.15 * N), replace=False)
        zw[cluster, c] = rng.uniform(0.12, 0.30, size=len(cluster))
        zd[c] = rng.integers(0, 2)
    cload = zw.sum(1)
    cscale = np.minimum(1.0, (0.95 - alpha) / np.maximum(cload, 1e-9))
    zw = zw * cscale[:, None]

    # diurnal modulation (drives the 6-window segmentation, as in real data)
    t = np.arange(T)
    hour = (t * dt_min / 60.0) % 24
    diurnal = 1.0 + 0.4 * np.sin(2 * np.pi * (t % 288) / 288.0 - np.pi / 3)

    by_target = {}
    for (j, i), (tau, beta) in true_edges.items(): by_target.setdefault(i, []).append((j, tau, beta))

    X = np.zeros((T, N)); z = np.zeros((C, T))
    z[:, 0] = rng.normal(0, zsd, C)
    eps_sd = 0.6
    for tt in range(1, T):
        u = diurnal[tt]
        z[:, tt] = zphi * z[:, tt - 1] + rng.normal(0, zsd, C)
        zc = (z[:, tt] * u)
        xt = X[tt - 1] * alpha
        for i, lst in by_target.items():
            acc = 0.0
            for (j, tau, beta) in lst:
                acc += beta * X[tt - tau, j]
            xt[i] += acc
        xt += zw @ zc
        xt += rng.normal(0, 1.0, N) * eps_sd * u
        X[tt] = xt
    # burn-in discard
    X = X[500:]
    tod = (hour[500:] / 24.0).astype(np.float32)
    return X, tod, true_edges, zw

# ------------------------------- pipeline -----------------------------------
def run_pipeline(X, tod, rng):
    T, N = X.shape
    lags = list(range(1, L_MAX + 1))
    per_window = []
    for wname, lo, hi in WINDOWS:
        m = ((tod * 24.0) >= lo) & ((tod * 24.0) < hi)
        Xw = X[m]
        mu = Xw.mean(0); sd = Xw.std(0); sd[sd < 1e-8] = 1
        Xs = (Xw - mu) / sd
        Tw = Xs.shape[0]
        C = cross_corr_lags(Xs, lags)
        kb = np.abs(C).argmax(axis=0)
        rho = np.take_along_axis(C, kb[None], axis=0)[0]
        tau = (kb + 1).astype(float)
        dof = max(Tw - 2, 3)
        with np.errstate(divide="ignore", invalid="ignore"):
            tstat = rho * np.sqrt(dof / np.maximum(1 - rho ** 2, 1e-12))
        from scipy import stats as sps
        p = 2 * sps.t.sf(np.abs(tstat).ravel(), df=dof).reshape(N, N)
        q = bh_fdr(p.ravel()).reshape(N, N)
        per_window.append(dict(Tw=Tw, rho=rho, tau=tau, q=q))
    # aggregate (sample-size weighted), same as real pipeline
    Tw_tot = sum(w["Tw"] for w in per_window)
    rhoA = sum(w["Tw"] * w["rho"] for w in per_window) / Tw_tot
    tauA = sum(w["Tw"] * w["tau"] for w in per_window) / Tw_tot
    sigA = sum(w["Tw"] * (w["q"] < FDR_Q) for w in per_window) / Tw_tot
    mask_any = sigA > 0.5                      # significant in the majority of windows
    np.fill_diagonal(mask_any, False)
    return rhoA, tauA, mask_any, per_window

def ci_screen(X, mask, tauA, A_road, rhoA):
    """Reproduce the conditional-independence confound screen on synthetic data."""
    from scipy.sparse import csgraph
    D, pred = csgraph.shortest_path(csr_matrix(A_road), method="D", unweighted=True,
                                    return_predecessors=True)
    def path_nodes(j, i, cap=3):
        if not np.isfinite(D[j, i]) or D[j, i] < 2: return []
        nodes, cur = [], i
        while cur != j and len(nodes) < 64:
            cur = pred[j, cur]
            if cur < 0: return []
            if cur != j: nodes.append(int(cur))
        return nodes[:cap]
    T, N = X.shape
    mu, sd = X.mean(0), X.std(0); sd[sd < 1e-8] = 1
    S = (X - mu) / sd
    conf = np.zeros_like(mask)
    ei, ej = np.nonzero(mask)
    for i, j in zip(ei, ej):
        tlag = max(int(round(tauA[i, j])), 1)
        xi = S[tlag:, i]; xj = S[:-tlag, j]
        inter = path_nodes(j, i)
        if not inter: continue
        M = np.column_stack([S[:-tlag, k] for k in inter])
        Xd = np.column_stack([np.ones(len(xi)), M])
        bi, *_ = np.linalg.lstsq(Xd, xi, rcond=None)
        bj, *_ = np.linalg.lstsq(Xd, xj, rcond=None)
        pr = np.corrcoef(xi - Xd @ bi, xj - Xd @ bj)[0, 1]
        if abs(pr) < CI_DROP_RATIO * abs(rhoA[i, j]):
            conf[i, j] = True
    return conf

def main():
    rng = np.random.default_rng(SEED)
    print("generating synthetic data ...", flush=True)
    X, tod, GT, zw = generate(T=9600, rng=rng)
    N = X.shape[1]
    print(f"X: {X.shape}, true edges: {len(GT)}", flush=True)
    A_road = np.asarray(np.load(os.path.join(HERE, "data", "adj_mx.npy")), dtype=float)

    rhoA, tauA, mask_any, per_window = run_pipeline(X, tod, rng)
    print("pipeline done; raw significant pairs:", int(mask_any.sum()), flush=True)
    conf = ci_screen(X, mask_any, tauA, A_road, rhoA)
    print("confounded flagged:", int(conf.sum()), flush=True)

    gt = set(GT.keys())
    def metrics(m):
        m = m.copy(); np.fill_diagonal(m, False)
        disc = {(int(j), int(i)) for i, j in zip(*np.nonzero(m))}
        tp = len(disc & gt); fp = len(disc - gt); fn = len(gt - disc)
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        matched = disc & gt
        dtau = [abs(tauA[i, j] - GT[(j, i)][0]) for (j, i) in matched]
        exact = float(np.mean([1.0 if d < 0.5 else 0.0 for d in dtau])) if dtau else 0.0
        med = float(np.median(dtau)) if dtau else float("nan")
        return dict(precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4),
                    tp=tp, fp=fp, fn=fn, median_abs_dtau=med, exact_lag_pct=round(100 * exact, 1))

    K = 10
    keep = np.zeros_like(mask_any)
    for i in range(N):
        js = np.nonzero(mask_any[i] & ~conf[i])[0]
        if len(js) > K: js = js[np.argsort(-np.abs(rhoA[i, js]))[:K]]
        keep[i, js] = True

    variants = {
        "V1_p_only":        metrics(mask_any),
        "V2_rho_only":      metrics(np.abs(rhoA) > RHO_THRESH),
        "V3_full_admission": metrics(mask_any & (np.abs(rhoA) > RHO_THRESH)),
        "V4_plus_confound_screen": metrics(mask_any & (np.abs(rhoA) > RHO_THRESH) & ~conf),
        "V5_deployed_capK10": metrics(keep & (np.abs(rhoA) > RHO_THRESH)),
    }
    out = dict(seed=SEED, n_nodes=int(N), n_true_edges=len(gt),
               confounders_flagged=int(conf.sum()),
               variants=variants)
    with open(os.path.join(RES, f"synthetic_stats_s{SEED}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(variants, indent=2))
    return out

if __name__ == "__main__":
    main()
