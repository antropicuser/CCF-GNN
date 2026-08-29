"""Generate ALL paper figures from REAL result artifacts (JSON/NPZ/CSV in this workspace).
No synthetic numbers: every point traces to a saved run or the audited METR-LA graph.
Outputs -> paper/figures/*.pdf  (+ 'FIGURES DONE' at the end)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data")
FIGS = os.path.abspath(os.path.join(HERE, "figures"))
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_CCF, C_GWN, C_HA = "#1f77b4", "#d62728", "#7f7f7f"

# ---------------------------------------------------------------- fig1: pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")

    def box(x, y, w, h, title, lines, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    fc=fc, ec="#444444", lw=0.8))
        ax.text(x + w/2, y + h - 0.28, title, ha="center", va="top",
                fontsize=8, fontweight="bold")
        ax.text(x + 0.15, y + h - 0.62, "\n".join(lines), ha="left", va="top", fontsize=6.6)

    box(0.15, 1.15, 2.9, 3.1, "A. Offline CCF audit",
        ["speed series, 207 sensors", r"pairwise CCF $\rightarrow$ $(\hat\tau_{ij},\hat\rho_{ij})$",
         "regime-wise windows (W=6)", "", "24,179 raw directed edges"],
        "#eef3fb")
    box(3.55, 1.15, 2.9, 3.1, "B. Statistical screening",
        ["space--time feasibility", r"($\hat\tau \leq d_{ij}/v_\mathrm{min}$)",
         "effect size + BH--FDR", "path-based confound screen", "",
         "16,181 validated (feas. 22.7%)"],
        "#edf6ed")
    box(6.95, 1.15, 2.9, 3.1, "C. CCF-GNN forecast",
        [r"fixed prior $\mathcal{A}$ + per-edge gates", r"lag-aligned propagation ($K_g{=}10$)",
         r"decoder $\rightarrow$ 15/30/60 min", r"top-$K$ validated tuples", r"$\rightarrow$ template explanations"],
        "#fdf1ec")

    for x0 in (3.12, 6.52):
        ax.add_patch(FancyArrowPatch((x0, 2.7), (x0 + 0.38, 2.7),
                     arrowstyle="-|>", mutation_scale=10, color="#333333", lw=1.0))
    ax.text(5.0, 0.62, "all statistics computed on the training split only; the graph is fixed at test time",
            ha="center", fontsize=6.4, style="italic", color="#444444")
    ax.text(5.0, 4.42, "edge attributes: lag $\\hat\\tau_{ij}$ (aligned convolution) and confidence $\\hat\\rho_{ij}$ (gate initialisation)",
            ha="center", fontsize=6.6, color="#333333")
    fig.savefig(os.path.join(FIGS, "fig1_pipeline.pdf")); plt.close(fig)

# ------------------------------------------------------- fig2: lag + distance-lag
def fig_anatomy():
    z = np.load(os.path.join(RES, "ccf_graph.npz"), allow_pickle=True)
    mask, val, conf, tau = z["mask"], z["validated"], z["confounded"], z["tau_agg"]
    audit = json.load(open(os.path.join(RES, "distance_lag_audit.json")))

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2), layout="constrained")
    ax = axes[0]
    tv = tau[mask & val]; tc = tau[mask & conf]
    STEP_MIN = 5.0  # one lag step = one 5-min METR-LA bucket
    tv = STEP_MIN * tv; tc = STEP_MIN * tc
    med = STEP_MIN * float(np.median(tau[mask]))
    bins = np.arange(0, 66, 3)
    ax.hist(tv, bins=bins, color=C_CCF, alpha=0.75, label=f"validated (n={len(tv):,})", density=True)
    ax.hist(tc, bins=bins, color=C_GWN, alpha=0.55, label=f"confounded (n={len(tc):,})", density=True)
    ax.axvline(med, color="#333333", lw=1.0, ls="--")
    ax.text(0.98, 0.70, f"median $\\hat\\tau$ = {med:.1f} min",
            transform=ax.transAxes, ha="right", fontsize=6.8)
    ax.set_xlabel(r"estimated lag $\hat\tau_{ij}$ (min)"); ax.set_ylabel("density")
    ax.set_title("(a) Lag estimates (raw CCF graph)"); ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    import csv
    id2idx = {s: k for k, s in enumerate(json.load(
        open(os.path.join(DATA, "adj_mx_mapping.json")))["sensor_ids"])}
    xs_v, ys_v, xs_c, ys_c = [], [], [], []
    with open(os.path.join(DATA, "distances.csv")) as f:
        for row in csv.DictReader(f):
            i, j = id2idx.get(row["from"]), id2idx.get(row["to"])
            if i is None or j is None or not mask[i, j]: continue
            d_km = float(row["cost"]) / 1000.0
            if val[i, j]: xs_v.append(STEP_MIN * tau[i, j]); ys_v.append(d_km)
            elif conf[i, j]: xs_c.append(STEP_MIN * tau[i, j]); ys_c.append(d_km)
    ax.scatter(xs_v, ys_v, s=2, alpha=0.25, color=C_CCF, rasterized=True,
               label=f"validated ($\\rho_s$={audit['validated']['spearman_rho']:.3f})")
    ax.scatter(xs_c, ys_c, s=2, alpha=0.20, color=C_GWN, rasterized=True,
               label=f"confounded ($\\rho_s$={audit['confounded']['spearman_rho']:.3f})")
    ax.set_xlabel(r"estimated lag $\hat\tau_{ij}$ (min)"); ax.set_ylabel("road distance $d_{ij}$ (km)")
    ax.set_title("(b) Lag vs. road distance, by verdict")
    ax.legend(frameon=True, facecolor="white", framealpha=0.75, edgecolor="none", loc="upper right")
    fig.savefig(os.path.join(FIGS, "fig2_anatomy.pdf")); plt.close(fig)

# ------------------------------------------------------------- fig3: sensor map
def fig_map():
    z = np.load(os.path.join(RES, "ccf_graph.npz"), allow_pickle=True)
    val, rho = z["validated"], z["rho_agg"]
    loc = {}
    with open(os.path.join(DATA, "sensor_locations.csv")) as f:
        import csv
        for row in csv.DictReader(f):
            loc[row["sensor_id"]] = (float(row["longitude"]), float(row["latitude"]))
    ids = json.load(open(os.path.join(DATA, "adj_mx_mapping.json")))["sensor_ids"]
    P = np.array([loc[s] for s in ids])

    fig, ax = plt.subplots(figsize=(3.4, 3.4), layout="constrained")
    segs, colors = [], []
    ii, jj = np.where(val)
    for i, j in zip(ii, jj):
        segs.append([P[i], P[j]]); colors.append(rho[i, j])
    lc = LineCollection(segs, colors=plt.cm.viridis(np.array(colors) * 8), alpha=0.05, lw=0.4)
    ax.add_collection(lc)
    indeg = val.sum(0)
    sc = ax.scatter(P[:, 0], P[:, 1], c=indeg, s=6 + indeg, cmap="viridis", lw=0, alpha=0.95)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7); cb.set_label("validated in-degree", fontsize=7)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_aspect("equal")
    fig.savefig(os.path.join(FIGS, "fig3_map.pdf")); plt.close(fig)

# ----------------------------------------------------------- fig4: medium bars
def fig_medium():
    runs = json.load(open(os.path.join(RES, "runs.json")))
    HZ = ["15min", "30min", "60min"]; hz_lab = ["15 min", "30 min", "60 min"]
    def stats(mode):
        e = runs[mode]
        m = np.array([[x[h]["MAE"] for h in HZ] for x in e])
        return m.mean(0), m.std(0, ddof=1) if len(e) > 1 else np.zeros(3)
    ha_m, _ = stats("HA"); gw_m, gw_s = stats("gwnet"); cc_m, cc_s = stats("full")

    x = np.arange(3); w = 0.26
    fig, ax = plt.subplots(figsize=(4.6, 2.6), layout="constrained")
    for k, (m, s, c, lab, off) in enumerate([
            (ha_m, None, C_HA, "Historical Average", -w),
            (gw_m, gw_s, C_GWN, "Graph WaveNet", 0.0),
            (cc_m, cc_s, C_CCF, "CCF-GNN", w)]):
        ax.bar(x + off, m, w, yerr=s, capsize=2.5, color=c, alpha=0.88, label=lab,
               error_kw=dict(lw=0.9))
        for xi, v in zip(x + off, m):
            ax.text(xi, v + 0.15, f"{v:.2f}", ha="center", fontsize=6.8)
    for xi, (a, b) in zip(x, zip(gw_m, cc_m)):
        pct = (b - a) / a * 100
        ax.text(xi + w, b - 0.45, f"{pct:+.1f}%", ha="center", va="top", fontsize=6.6,
                color="white", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(hz_lab); ax.set_ylabel("masked MAE (mph)")
    ax.set_ylim(0, 10.2)
    fig.legend(frameon=False, loc="outside upper center", ncol=3, columnspacing=1.2, handlelength=1.2)
    fig.savefig(os.path.join(FIGS, "fig4_medium.pdf")); plt.close(fig)

# ------------------------------------------------------------ fig5: robustness
def fig_robust():
    r = json.load(open(os.path.join(RES, "robustness.json")))["results"]
    lv = [0, 5, 10, 15]
    def get(model):
        m = np.array([r[model][f"{l}%"][0] for l in lv])
        s = np.array([r[model][f"{l}%"][1] for l in lv])
        rel = np.array([r[model][f"{l}%_rel"][0] for l in lv[1:]])
        rels = np.array([r[model][f"{l}%_rel"][1] for l in lv[1:]])
        return m, s, rel, rels
    cm, cs, cr, crs = get("ccf_gnn"); gm, gs, gr, grs = get("gwnet")

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3), layout="constrained")
    ax = axes[0]
    ax.errorbar(lv, cm, yerr=cs, marker="o", ms=3.5, lw=1.2, color=C_CCF, label="CCF-GNN", capsize=2.5)
    ax.errorbar(lv, gm, yerr=gs, marker="s", ms=3.5, lw=1.2, color=C_GWN, label="Graph WaveNet", capsize=2.5)
    ax.set_xlabel("masked nodes (%)"); ax.set_ylabel("masked MAE (mph)")
    ax.set_title("(a) Absolute MAE under masking"); ax.set_xticks(lv)

    ax = axes[1]; x = np.arange(3); w = 0.34
    ax.bar(x - w/2, cr, w, yerr=crs, capsize=2.5, color=C_CCF, alpha=0.88, label="CCF-GNN", error_kw=dict(lw=0.9))
    ax.bar(x + w/2, gr, w, yerr=grs, capsize=2.5, color=C_GWN, alpha=0.88, label="Graph WaveNet", error_kw=dict(lw=0.9))
    ax.set_xticks(x); ax.set_xticklabels([f"{l}%" for l in lv[1:]])
    ax.set_xlabel("masked nodes (%)"); ax.set_ylabel("rel. MAE increase (%)")
    ax.set_title("(b) Relative degradation")
    handles, labels_ = ax.get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, loc="outside upper center", ncol=2,
               columnspacing=1.4, handlelength=1.2)
    fig.savefig(os.path.join(FIGS, "fig5_robust.pdf")); plt.close(fig)

# ------------------------------------------------------------- fig6: synthetic
def fig_synthetic():
    s = json.load(open(os.path.join(RES, "synthetic_stats.json")))["aggregated"]
    keys = list(s.keys())
    labs = {"V1_p_only": "V1\n$p$-only", "V2_rho_only": "V2\n$\\rho$-only",
            "V3_full_admission": "V3\n+full adm.", "V4_plus_confound_screen": "V4\n+confound",
            "V5_deployed": "V5\n+regime (deployed)"}
    P  = [s[k]["precision"][0] for k in keys];  Ps  = [s[k]["precision"][1] for k in keys]
    R  = [s[k]["recall"][0] for k in keys];     Rs  = [s[k]["recall"][1] for k in keys]
    F1 = [s[k]["f1"][0] for k in keys]
    x = np.arange(len(keys)); w = 0.36
    fig, ax = plt.subplots(figsize=(4.8, 2.6), layout="constrained")
    ax.bar(x - w/2, P, w, yerr=Ps, capsize=2.5, color="#7fa8d9", label="precision", error_kw=dict(lw=0.9))
    ax.bar(x + w/2, R, w, yerr=Rs, capsize=2.5, color="#e8a06a", label="recall", error_kw=dict(lw=0.9))
    ax.plot(x, F1, "kD", ms=4, label="$F_1$")
    for xi, f in zip(x, F1):
        ax.text(xi, f + 0.045, f"{f:.2f}", ha="center", fontsize=6.6)
    ax.set_xticks(x); ax.set_xticklabels([labs.get(k, k) for k in keys], fontsize=6.8)
    ax.set_ylim(0, 1.12); ax.set_ylabel("score")
    fig.legend(frameon=False, loc="outside upper center", ncol=3, columnspacing=1.2, handlelength=1.2)
    fig.savefig(os.path.join(FIGS, "fig6_synthetic.pdf")); plt.close(fig)

if __name__ == "__main__":
    fig_pipeline(); print("fig1 pipeline ok")
    fig_anatomy();  print("fig2 anatomy ok")
    fig_map();      print("fig3 map ok")
    fig_medium();   print("fig4 medium ok")
    fig_robust();   print("fig5 robust ok")
    fig_synthetic();print("fig6 synthetic ok")
    print("FIGURES DONE")
