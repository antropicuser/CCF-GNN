"""Compact but faithful Graph WaveNet (Wu et al., IJCAI 2019) — pilot scale.

Architecture: gated dilated causal 1D convolutions (2 blocks x 4 layers, dilation 1..8)
with self-adaptive adjacency softmax(ReLU(E1 E2^T)), graph convolution after each gated
layer, skip connections, direct multi-horizon output (H=12). Reduced width d=32 to match
the CCF-GNN pilot budget exactly (same data subsampling, epochs, batch, seeds).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.lin = nn.Linear(cin, cout)
    def forward(self, X, A):
        # X: (B, C, N, T); A: (N,N) row-normalised (adaptive + I)
        Z = self.lin(X.permute(0, 2, 3, 1))          # (B,N,T,cout)
        Zp = Z.permute(0, 3, 1, 2)                   # (B,cout,N,T)
        return torch.einsum('ij,bcjt->bcit', A, Zp)  # (B,cout,N,T)

class GWLayer(nn.Module):
    def __init__(self, cin, cout, skip_ch, dilation):
        super().__init__()
        self.filt = nn.Conv2d(cin, cout, (1, 2), dilation=(1, dilation))
        self.gate = nn.Conv2d(cin, cout, (1, 2), dilation=(1, dilation))
        self.gcn  = GraphConv(cout, cout)
        self.res  = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.skip = nn.Conv2d(cout, skip_ch, 1)
        nn.init.xavier_uniform_(self.filt.weight); nn.init.zeros_(self.filt.bias)
        nn.init.xavier_uniform_(self.gate.weight); nn.init.zeros_(self.gate.bias)

    def forward(self, x, A):
        res = self.res(x)
        z = F.pad(x, (self.filt.dilation[1], 0))     # causal left-pad on time
        z = torch.tanh(self.filt(z)) * torch.sigmoid(self.gate(z))
        z = self.gcn(z, A)
        return res + z, self.skip(z[:, :, :, -1:])  # skip from latest time (keep 4D)

class GWNet(nn.Module):
    def __init__(self, N, d=32, H=12, emb=10, blocks=2, layers=4, skip_ch=64, dropout=0.3):
        super().__init__()
        self.N, self.d, self.H = N, d, H
        self.E1 = nn.Parameter(torch.randn(N, emb) * 0.1)
        self.E2 = nn.Parameter(torch.randn(emb, N) * 0.1)
        self.start = nn.Conv2d(1, d, 1)
        self.layers = nn.ModuleList([GWLayer(d if i == 0 else d, d, skip_ch, 2 ** (i % layers))
                                     for i in range(blocks * layers)])
        self.end1 = nn.Conv2d(skip_ch, 2 * d, 1)
        self.end2 = nn.Conv2d(2 * d, H, 1)
        self.drop = nn.Dropout(dropout)

    def adaptive_A(self):
        A = F.softmax(F.relu(self.E1 @ self.E2), dim=1)
        A = A + torch.eye(self.N, device=A.device)
        return A / A.sum(dim=1, keepdim=True).clamp(min=1e-6)

    def forward(self, x, H_steps=None):
        # x: (B, T, N) speeds (z-scored) -> (B, 1, N, T)
        B, T, N = x.shape
        h = self.start(x.transpose(1, 2).unsqueeze(1))
        A = self.adaptive_A()
        skip = 0
        for layer in self.layers:
            h, s = layer(h, A)
            skip = skip + s
        z = F.relu(skip)
        z = self.drop(z)
        z = self.end2(F.relu(self.end1(z)))          # (B, H, N)
        return z.squeeze(-1).permute(0, 2, 1)        # (B, N, H)
