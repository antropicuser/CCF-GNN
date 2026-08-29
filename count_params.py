"""Count trainable parameters of CCF-GNN at the paper configuration (d=64, N=207).

Usage: python count_params.py
Guards the paper's efficiency claims: never quote a parameter count without running this.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_graph
from models.ccf_gnn import CCF_GNN

g = load_graph(k_cap=10)
for d in (32, 64):
    m = CCF_GNN(g, d=d)
    total = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"d={d}: {total:,} trainable parameters "
          f"(hybrid gates: {m.beta.numel():,}; node embeddings: {m.E1.numel()+m.E2.numel():,})")
print("NOTE: baselines (GWNet ~0.47M, DCRNN ~0.31M in standard configs) must be counted from "
      "their official implementations under the same configuration -- do not reuse paper claims.")
