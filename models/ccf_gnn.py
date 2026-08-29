"""CCF-GNN model (pilot scale, faithful to the paper's methodology).

  * Hybrid adjacency:  A = M ⊙ [ alpha ⊙ A_CCF + (1 - alpha) ⊙ softmax(ReLU(E1 E2^T)) ],
    alpha = sigmoid(beta), beta initialised so alpha = 0.7 (validated) / 0.1 (confounded);
    outside the capped CCF mask the residual graph acts alone.
  * Time-aligned graph convolution via lag-grouped adjacency:
        h_i(t) = relu( Σ_l  (A_l @ W) h(t - l) + b ),   A_l[i,j] = A_ij * 1[tau_ij = l].
  * Encoder: 2 x [LagGCN -> GRU]; decoder: autoregressive instantaneous-GCN GRU + MLP head.
  * Lag alignment uses the aggregated lag table (window-specific tables remain an option).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LagGCN(nn.Module):
    """Time-aligned graph convolution with lag-grouped SPARSE propagation."""
    def __init__(self, din, dout, L=12):
        super().__init__()
        self.lin = nn.Linear(din, dout)
        self.L = L

    def forward(self, X, A_lags):
        # X: (B,T,N,din); A_lags: (L,N,N) differentiable lag-grouped adjacency
        H = self.lin(X)                       # (B,T,N,dout)
        B, T = X.shape[0], X.shape[1]
        out = torch.zeros_like(H)
        for l in range(1, self.L + 1):
            Al = A_lags[l - 1]
            if not torch.any(Al): continue
            P = torch.matmul(Al, H)           # (N,N)@(B,T,N,dout) broadcast; no big autograd saves
            out[:, l:] = out[:, l:] + P[:, :T - l]
        return F.relu(out)

class CCF_GNN(nn.Module):
    def __init__(self, graph, d=64, L=12, H=12):
        super().__init__()
        self.L, self.H, self.d = L, H, d
        N = graph["A_ccf"].shape[0]
        self.register_buffer("A_ccf", torch.as_tensor(graph["A_ccf"], dtype=torch.float32))
        self.register_buffer("maskM", torch.as_tensor(graph["mask"], dtype=torch.float32))
        mtype = torch.as_tensor(graph["mtype"], dtype=torch.float32)
        self.register_buffer("tau", torch.as_tensor(graph["tau"], dtype=torch.long))
        beta0 = torch.where(mtype > 0.9, torch.tensor(0.8473), torch.tensor(-2.1972))
        self.beta = nn.Parameter(beta0)                       # sigmoid -> 0.7 / 0.1
        self.E1 = nn.Parameter(torch.randn(N, 10) * 0.1)
        self.E2 = nn.Parameter(torch.randn(N, 10) * 0.1)
        self.gcn1, self.gru1 = LagGCN(1, d, L), nn.GRU(d, d, batch_first=True)
        self.gcn2, self.gru2 = LagGCN(d, d, L), nn.GRU(d, d, batch_first=True)
        self.gcn_dec = nn.Linear(d, d)
        self.gru_dec = nn.GRUCell(d, d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)  # start near 0
        # cache lag-grouped masks (structure constant during training)
        taus = self.tau
        self.register_buffer("lag_masks", torch.stack([taus == l for l in range(1, L + 1)]))  # (L,N,N)

    def hybrid_A(self):
        alpha = torch.sigmoid(self.beta) * self.maskM
        res = F.softmax(F.relu(self.E1 @ self.E2.T), dim=1)
        A = alpha * self.A_ccf + (1 - alpha) * res
        A = A + torch.eye(A.shape[0], device=A.device) * 0.5
        return A / A.sum(dim=1, keepdim=True).clamp(min=1e-6)   # random-walk normalisation

    def forward(self, x, H_steps=None):
        B, T, N = x.shape
        A = self.hybrid_A()
        A_lags = A.unsqueeze(0) * self.lag_masks          # (L,N,N), differentiable
        h = self.gcn1(x.unsqueeze(-1), A_lags)
        h, _ = self.gru1(h.reshape(B * N, T, -1)); h = h.reshape(B, T, N, self.d)
        h = self.gcn2(h, A_lags)
        h, _ = self.gru2(h.reshape(B * N, T, -1)); h = h.reshape(B, T, N, self.d)
        hd = h[:, -1]                                         # (B,N,d)
        prev = x[:, -1]                                       # (B,N) last observed
        outs = []
        steps = H_steps or self.H
        for k in range(steps):
            z = F.relu(self.gcn_dec(hd) + self.gcn_dec(prev.unsqueeze(-1).expand(-1, -1, self.d)))
            hd = self.gru_dec(z.reshape(B * N, self.d), hd.reshape(B * N, self.d)).reshape(B, N, self.d)
            y = self.head(hd).squeeze(-1)                     # (B,N)
            outs.append(y)
            prev = y
        return torch.stack(outs, dim=2)                       # (B,N,H)
