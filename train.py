"""Training / evaluation driver (pilot-scale defaults; full protocol via flags).

Examples
--------
Pilot (CPU, reduced):      python train.py --mode full --stride 4 --epochs 8 --seeds 1
Ablation (no time lags):   python train.py --mode nolag --stride 4 --epochs 8 --seeds 1
Baselines:                 python train.py --mode ha        (historical average)
All results -> results/runs.json  (consumed by make_tables.py)
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
from data_utils import load_split, zscore_fit, masked_metrics, load_graph
from models.ccf_gnn import CCF_GNN
from models.gwnet import GWNet

RES = os.path.join(os.path.dirname(__file__), "results")

def window_id_of(tod):
    """Map time-of-day in [0,1) -> window id 0..5 (paper Table: 6 diurnal windows)."""
    h = tod * 24.0
    bounds = [6, 8, 10, 15, 19, 24]
    return np.searchsorted(bounds, h)

def evaluate(model, X, Y, tod, mu, sd, wids):
    model.eval(); preds = np.zeros_like(Y, dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(X), 64):
            xb = torch.tensor((X[s:s+64] - mu) / sd, dtype=torch.float32).transpose(1, 2)  # (B,T,N)
            yb = torch.tensor(Y[s:s+64], dtype=torch.float32)
            wb = torch.tensor(wids[s:s+64])
            out = model(xb)
            preds[s:s+64] = out.numpy() * sd + mu
    out = {}
    for h, name in [(2, "15min"), (5, "30min"), (11, "60min")]:
        out[name] = masked_metrics(preds[:, :, h], Y[:, :, h])
    out["AVG"] = masked_metrics(preds, Y)
    return out

def run_seed(args, seed, g):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, Ytr, todtr = load_split("train", stride=args.stride)
    mu, sd = zscore_fit(Xtr)
    Xva, Yva, todva = load_split("val", stride=max(args.stride, 4))
    Xte, Yte, todte = load_split("test", stride=max(args.stride, 4))
    wtr, wva, wte = window_id_of(todtr), window_id_of(todva), window_id_of(todte)

    if args.mode == "ha":
        # historical average by (window_id, node) from train
        N = Xtr.shape[1]; table = np.zeros((6, N)); cnt = np.zeros((6, 1))
        for w in range(6):
            sel = wtr == w
            if sel.sum() == 0: continue
            # mean of last observed step per node as HA proxy, horizon-constant
            table[w] = Xtr[sel][:, :, -1].mean(0); cnt[w] = sel.sum()
        preds = np.zeros_like(Yte)
        for w in range(6):
            sel = wte == w
            preds[sel] = table[w][None, :, None]
        out = {}
        for h, name in [(2, "15min"), (5, "30min"), (11, "60min")]:
            out[name] = masked_metrics(preds[:, :, h], Yte[:, :, h])
        out["AVG"] = masked_metrics(preds, Yte)
        return out, None

    if args.mode == "gwnet":
        model = GWNet(Xtr.shape[1], d=args.d)
    else:
        model = CCF_GNN(g, d=args.d)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    if args.mode == "gwnet":
        pass
    elif args.mode == "nolag":   # ablation: force all lags to 1 (instantaneous alignment)
        model.tau.data = torch.ones_like(model.tau.data)
        model.lag_masks = torch.stack([model.tau == l for l in range(1, model.L + 1)]).to(model.tau.device)
    if args.mode == "nohybrid":  # pure CCF graph (alpha = 1)
        model.beta.data = torch.full_like(model.beta.data, 8.0)

    best_va, best_state, t0 = float("inf"), None, time.time()
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(Xtr))
        tot = 0.0
        for s in range(0, len(Xtr), args.batch):
            bi = perm[s:s+args.batch]
            xb = torch.tensor((Xtr[bi] - mu) / sd, dtype=torch.float32).transpose(1, 2)  # (B,T,N)
            yraw = torch.tensor(Ytr[bi], dtype=torch.float32)
            yb = (yraw - mu) / sd
            mask = (yraw > 0).float()
            out = model(xb)
            loss = (torch.abs(out - yb) * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tot += float(loss.detach())
        va = evaluate(model, Xva[:512], Yva[:512], todva[:512], mu, sd, wva[:512])["AVG"]["MAE"]
        if va < best_va:
            best_va = va; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"seed{seed} ep{ep+1}/{args.epochs} loss={tot/max(1,(len(Xtr)//args.batch)):.4f} vaMAE={va:.3f}", flush=True)
    model.load_state_dict(best_state)
    te = evaluate(model, Xte, Yte, todte, mu, sd, wte)
    te["_train_minutes"] = round((time.time() - t0) / 60, 1)
    te["_params"] = int(sum(p.numel() for p in model.parameters()))
    return te, best_state

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full", choices=["full", "nolag", "nohybrid", "nohybridgraph", "ha", "gwnet"])
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds (use >=10 for the paper)")
    ap.add_argument("--seed0", type=int, default=0, help="seed offset (seed = seed0+1..seed0+seeds)")
    args = ap.parse_args()

    runs_path = os.path.join(RES, "runs.json")
    runs = json.load(open(runs_path)) if os.path.exists(runs_path) else {}
    if args.mode == "ha":
        out, _ = run_seed(args, 0, None)
        runs.setdefault("HA", []).append(out)
        json.dump(runs, open(runs_path, "w"), indent=2)
        print("HA:", json.dumps(out, indent=2)); return

    g = load_graph(k_cap=10)
    print("graph edges (capped):", int(g["mask"].sum()), "alpha-init validated:", int((g["mtype"] == 1).sum()))
    allr = []
    for seed in range(args.seed0 + 1, args.seed0 + args.seeds + 1):
        te, _ = run_seed(args, seed, g)
        allr.append(te)
        print(f"== seed {seed} test:", json.dumps({k: v for k, v in te.items() if not k.startswith('_')}, indent=1))
    runs.setdefault(args.mode, []).extend(allr)
    json.dump(runs, open(runs_path, "w"), indent=2)

if __name__ == "__main__":
    main()
