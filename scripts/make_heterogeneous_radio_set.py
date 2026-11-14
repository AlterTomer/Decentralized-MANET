"""
Heterogeneous-radio test set.

Starting from the R2 main test set (|V| = 10, p = 0.3, 500 connected networks), each node supports
only a random subset of the B channels: every channel is supported independently with probability
SUPPORT_PROB, and each node supports at least one channel. As in the system model, a channel that
is not supported by node i is represented by zero coefficients, h_{i->j}^{(b)} = h_{j->i}^{(b)} = 0
for all j, so a link keeps channel b only if both of its nodes support b, and a link without any
common channel is removed from the topology. Support patterns are redrawn (seeded) until the
resulting network is connected, as assumed by the system model.

Output: torch .pt dict with 'H_all' [S,B,n,n] complex, 'support' [S,n,B] bool, 'p_vec' [S],
loadable through generate_graph_data(..., channel_path=<.pt>).
"""
import os
from pathlib import Path

import mat73
import numpy as np
import torch

QC = Path(os.path.join(os.environ.get("MANET_DATA_ROOT", r"C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET"), r"Quadriga channels"))
SRC = QC / "Test_H_seed_90210_p_0p3_n_10_B_6_samples_500.mat"
SUPPORT_PROB = 2.0 / 3.0
SEED = 4242
MAX_TRIES = 1000


def connected(A):
    n = A.shape[0]
    seen, stack = {0}, [0]
    while stack:
        u = stack.pop()
        for v in np.flatnonzero(A[u]):
            if int(v) not in seen:
                seen.add(int(v))
                stack.append(int(v))
    return len(seen) == n


m = mat73.loadmat(str(SRC))
H = np.asarray(m["H_all"]).astype(np.complex64)          # [S,B,n,n]
p_vec = np.asarray(m.get("p_vec", np.full(H.shape[0], 0.3))).ravel()
S, B, n, _ = H.shape
rng = np.random.default_rng(SEED)

H_out = np.zeros_like(H)
support = np.zeros((S, n, B), dtype=bool)
redraws = []
for s in range(S):
    for tries in range(1, MAX_TRIES + 1):
        sup = rng.random((n, B)) < SUPPORT_PROB
        empty = ~sup.any(axis=1)
        sup[empty, rng.integers(B, size=int(empty.sum()))] = True      # at least one channel per node
        common = sup[:, None, :] & sup[None, :, :]                       # [n,n,B]
        Hs = H[s] * np.transpose(common, (2, 0, 1))                      # zero unsupported channels
        A = np.abs(Hs).sum(axis=0) > 0
        if connected(A):
            break
    else:
        raise RuntimeError(f"network {s}: no connected support pattern in {MAX_TRIES} tries")
    H_out[s], support[s] = Hs, sup
    redraws.append(tries - 1)

A0 = np.abs(H).sum(axis=1) > 0
A1 = np.abs(H_out).sum(axis=1) > 0
lb0 = (np.abs(H) > 0).sum(axis=(1, 2, 3)) / 2
lb1 = (np.abs(H_out) > 0).sum(axis=(1, 2, 3)) / 2
print(f"channels supported per node: {support.sum(axis=2).mean():.2f} of {B} "
      f"(min {support.sum(axis=2).min()}, max {support.sum(axis=2).max()})")
print(f"links kept: {A1.sum() / A0.sum():.1%} | link-channel pairs kept: {lb1.sum() / lb0.sum():.1%} "
      f"| networks needing a redraw: {np.mean(np.array(redraws) > 0):.1%}")

out = QC / "Test_H_HETRADIO_seed_90210_p_0p3_n_10_B_6_samples_500.pt"
if out.exists():
    raise FileExistsError(f"{out} exists; delete it first to regenerate.")
torch.save({"H_all": torch.from_numpy(H_out), "support": torch.from_numpy(support),
            "p_vec": torch.from_numpy(p_vec), "source": str(SRC)}, out)
print(f"saved {out}")
