"""
Heterogeneous-radio TRAINING set.

Same procedure as scripts/make_heterogeneous_radio_set.py (the test set), applied to the training
channels Train_H_seed_48312_n_10_B_6_samples_2000.mat with its own seed: every node supports each of
the B channels independently with probability SUPPORT_PROB (at least one channel per node), a link
keeps channel b only if both of its nodes support b, and a link without a common channel is removed.
Unlike the test set, the training networks are not all connected, so the support pattern is redrawn
until it does not change the connected components of the network (in particular, a connected network
stays connected). The test set uses seed 4242 on independently generated test channels, so the two
sets share no network.

Output: Train_H_HETRADIO_seed_48312_n_10_B_6_samples_2000.pt with 'H_all' [S,B,n,n] complex,
'support' [S,n,B] bool, 'p_vec' [S], loadable by generate_graph_data and by MANET_FFN/main.py.
"""
import os
from pathlib import Path

import mat73
import numpy as np
import torch

QC = Path(os.path.join(os.environ.get("MANET_DATA_ROOT", r"C:\Users\alter\OneDrive\Desktop\PhD\Decentralized MANET"), r"Quadriga channels"))
SRC = QC / "Train_H_seed_48312_n_10_B_6_samples_2000.mat"
OUT = QC / "Train_H_HETRADIO_seed_48312_n_10_B_6_samples_2000.pt"
SUPPORT_PROB = 2.0 / 3.0          # as in the test set: on average 4 of the 6 channels per node
SEED = 4343
MAX_TRIES = 1000


def components(A):
    """Connected-component label of every node."""
    n = A.shape[0]
    lab = -np.ones(n, dtype=int)
    c = 0
    for s in range(n):
        if lab[s] >= 0:
            continue
        lab[s], stack = c, [s]
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(A[u]):
                if lab[v] < 0:
                    lab[v] = c
                    stack.append(int(v))
        c += 1
    return lab


def same_partition(a, b):
    return all(len(set(b[a == k])) == 1 for k in set(a)) and len(set(a)) == len(set(b))


if OUT.exists():
    raise FileExistsError(f"{OUT} exists; delete it first to regenerate.")
m = mat73.loadmat(str(SRC))
H = np.asarray(m["H_all"]).astype(np.complex64)          # [S,B,n,n]
p_vec = np.asarray(m.get("p_vec", np.full(H.shape[0], 0.3))).ravel()
S, B, n, _ = H.shape
rng = np.random.default_rng(SEED)

H_out = np.zeros_like(H)
support = np.zeros((S, n, B), dtype=bool)
redraws = []
for s in range(S):
    A0 = np.abs(H[s]).sum(axis=0) > 0
    np.fill_diagonal(A0, False)
    lab0 = components(A0)
    for tries in range(1, MAX_TRIES + 1):
        sup = rng.random((n, B)) < SUPPORT_PROB
        empty = ~sup.any(axis=1)
        sup[empty, rng.integers(B, size=int(empty.sum()))] = True      # at least one channel per node
        common = sup[:, None, :] & sup[None, :, :]                       # [n,n,B]
        Hs = H[s] * np.transpose(common, (2, 0, 1))                      # zero unsupported channels
        A = np.abs(Hs).sum(axis=0) > 0
        np.fill_diagonal(A, False)
        if same_partition(lab0, components(A)):
            break
    else:
        raise RuntimeError(f"network {s}: no support pattern keeping its components in {MAX_TRIES} tries")
    H_out[s], support[s] = Hs, sup
    redraws.append(tries - 1)

A0 = np.abs(H).sum(axis=1) > 0
A1 = np.abs(H_out).sum(axis=1) > 0
lb0 = (np.abs(H) > 0).sum(axis=(1, 2, 3)) / 2
lb1 = (np.abs(H_out) > 0).sum(axis=(1, 2, 3)) / 2
print(f"channels supported per node: {support.sum(axis=2).mean():.2f} of {B}")
print(f"links kept: {A1.sum() / A0.sum():.1%} | link-channel pairs kept: {lb1.sum() / lb0.sum():.1%} "
      f"| networks needing a redraw: {np.mean(np.array(redraws) > 0):.1%}")
torch.save({"H_all": torch.from_numpy(H_out), "support": torch.from_numpy(support),
            "p_vec": torch.from_numpy(p_vec), "source": str(SRC)}, OUT)
print(f"saved {OUT}")
