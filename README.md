# MANET-GNN: Learned Decentralized Optimization of Power Allocation in Multi-Channel MANETs

> Official implementation of **MANET-GNN**, a decentralized graph neural network framework for learned power allocation in dynamic multi-channel MANETs.

This repository accompanies the paper:

**Tomer Alter, Nir Shlezinger, and Michael Segal**
*MANET-GNN: Learned Decentralized Optimization of Power Allocation in Multi-Channel MANETs* - Extended journal version under review.

---

## Overview

MANET-GNN is a decentralized learned optimization framework for power allocation in dynamic multi-hop multi-channel MANETs. The framework supports multiple communication paradigms within a unified formulation:

* Unicast
* Multicast
* Multicommodity
* Convergecast
* Many-to-Many

The method combines:

* Message-passing Graph Neural Networks (GNNs)
* Learned decentralized optimization
* Multi-channel routing-aware power allocation
* Robust training under noisy CSI

Unlike centralized optimization approaches, MANET-GNN operates using:

* Local neighborhood information only
* Limited message-passing rounds
* Distributed inference
* Fixed communication latency

The framework was evaluated under:

* QuaDRiGa frequency-selective channels (Rayleigh fading also supported)
* Full CSI and noisy (estimated) CSI regimes

and demonstrated near-centralized performance across all communication frameworks.

> **Reproducing the paper:** see **[docs/REPRODUCE.md](docs/REPRODUCE.md)** (data generation, training,
> benchmarks, and every additional study, with the exact configs in [config/paper/](config/paper)).
> **What's new in the second revision:** see [CHANGELOG.md](CHANGELOG.md).

---

## Key Features

### Unified Multi-Framework MANET Optimization

A single formulation supporting:

| Framework | Description    | Config `mode`  |
| --------- | -------------- | -------------- |
| F1        | Unicast        | `single`       |
| F2        | Multicast      | `multicast`    |
| F3        | Multicommodity | `multi`        |
| F4        | Convergecast   | `converge`     |
| F5        | Many-to-Many   | `multiunicast` |

> F5 (**Many-to-Many** in the paper) is the code's `multiunicast` mode — multiple independent
> transmitter/receiver pairs.

---

### Decentralized Message-Passing Architecture

MANET-GNN performs distributed optimization using local neighbor exchanges only.

Each layer performs:

1. Edge-aware message encoding
2. Neighborhood aggregation
3. Learned power refinement

The architecture explicitly limits the number of message-passing rounds to satisfy latency constraints.

---

### Learned Optimization

The GNN is trained as a distributed optimizer using an unsupervised loss derived directly from the centralized optimization objective.

Training includes:

* End-to-end throughput maximization
* Monotonic optimization regularization
* Noisy-CSI-aware training

---

### Generalization Across Topologies

The model generalizes across:

* Different graph sizes
* Different connectivity patterns
* Different channel realizations
* Different SNR regimes

---

## Repository Structure

```text
Decentralized-MANET/
│
├── MANET_FFN/                 # FFN baseline (B7): model, dataset, training (main.py)
├── Multicast/                 # Multicast subgraph and objective utilities
├── Multicommodity/            # Multicommodity objective utilities
├── config/                    # Example configs + config/paper/: the configs of every reported result
├── datasets/                  # Dataset generation and loading utilities
├── docs/                      # DESIGN.md (technical notes), REPRODUCE.md (reproduction guide)
├── matlab/                    # QuaDRiGa channel and test-set generation
├── models/                    # MANET-GNN architectures and layers, aux function for training and evaluation
├── scripts/                   # Training, benchmarking, and study scripts
├── utils/                     # Graph, routing, normalization, and helper functions
├── visualization/             # Visualization and graphing utilities
├── CHANGELOG.md
├── requirements.txt
└── README.md
```

---

## System Model

> **Full technical details** — the complete system model, objective derivation, MANET-GNN
> architecture (with equations), training procedure, and the key algorithmic/engineering design
> decisions are documented in **[docs/DESIGN.md](docs/DESIGN.md)**. The sections below are a summary.

We consider a dynamic multi-hop multi-channel MANET represented by an undirected graph:

```math
G = (V, E)
```

where:

* Nodes represent mobile devices
* Edges represent reciprocal wireless links
* Each link contains (B) orthogonal frequency channels

Power entries are **transmission amplitudes**, and links that reuse a channel generate co-channel
interference, so the achievable rate over link ((i,j)) and channel (b) is an **SINR** expression:

```math
R_{i \rightarrow j,k}^{(b)} = \log_2\left(1 + \frac{|h_{i\rightarrow j}^{(b)}|^2\,[P]_{b,k,i,j}^{2}}{\sigma_b^2 + I_{i\rightarrow j}^{(b)}(P)}\right)
```

where $I_{i\rightarrow j}^{(b)}(P)$ is the aggregate co-channel interference from neighboring
transmitters (treating interference as noise). End-to-end rates take the bottleneck along a route and
**sum over bands**; the objective maximizes the minimum end-to-end rate across messages.

The framework jointly optimizes:

* Multi-hop routing
* Multi-channel power allocation
* Commodity assignment

subject to per-node power constraints.

---

## MANET-GNN Architecture

The architecture consists of:

1. Input encoding
2. Gated message-passing backbone
3. Edge-aware FiLM modulation
4. Distributed power decoders

Each GNN layer performs:

* Local message generation
* Neighbor aggregation
* Residual feature refinement

Final outputs include:

* Power allocation tensor
* Commodity assignment masks

The design follows the architecture described in Sections III-A and III-B of the paper.

---

## Training

MANET-GNN is trained using unsupervised learning.

The training objective directly maximizes the end-to-end communication rate:

```math
\max_P \min_k R_k^{E2E}(P)
```

The total loss combines:

* Rate maximization (max-min end-to-end throughput)
* Monotonic improvement regularization
* Edge-sparsity regularization for single-message frameworks (multi-message routing is concentrated
  by a candidate-based routing head instead)

Training setup (from the paper's experiments):

* AdamW optimizer (lr `5e-4`; `1e-3` for unicast; weight decay `3e-5`), cosine schedule, dropout `0.2`
* 3 gated GNN layers (`L = 6` communication rounds) for F3–F5, 2 gated layers (`L = 4`) for F1–F2
* QuaDRiGa frequency-selective channels (Rayleigh also supported)
* SNR range: 0–50 dB in 5 dB steps, with the SNR given to the model as an input feature
* 2000 training topologies (ER, `p = 0.3`; 1600 train / 400 validation), 10 nodes (6 for multicast), 6 bands

The per-framework settings are listed in [docs/REPRODUCE.md](docs/REPRODUCE.md#2-training).

---

## Benchmarks

The repository includes comparisons against the following baselines, evaluated by
`scripts/Optimizer_vs_GNN.py` across the full SNR sweep:

| Baseline                                   | Description                                                        |
| ------------------------------------------ | ------------------------------------------------------------------ |
| B1 Centralized Optimizer                   | AdamW over the full MANET, started from the Greedy-Split, Widest-Path, and MANET-GNN allocations; best iterate kept (reference) |
| B2 Equal-Split                             | Uniform power over all feasible outgoing links and bands           |
| B3 Centralized Greedy-Split (CGS)          | Shortest route, power split equally over its links and all bands   |
| B4 Decentralized Greedy-Split (DGS)        | Bellman–Ford shortest route (L/2 rounds), equal split              |
| B5 Centralized Widest-Path (CWP)           | Route and band with the strongest bottleneck (ties: fewest hops), single-band allocation |
| B6 Decentralized Widest-Path (DWP)         | Bellman–Ford max–min widest route                                  |
| B7 MANET-FFN                               | Topology-blind MLP over the CSI of all links                       |

All methods are scored through the identical **interference-aware** rate objective (same routing
candidate set and band aggregation), so learned and heuristic allocations are compared on equal footing.

### Confidence intervals

Reported results are averaged over **500 independent test topologies** (10 nodes, 6 bands; `K = 4`
messages for multi-message frameworks), with **95% confidence intervals** (normal-approximation,
`±1.96·SEM`). The benchmark stores per-network spread (`results["sem"]`) and the test-set size
(`results["n_test"]`) alongside the means, and `plot_mean_rate_vs_snr` renders error bars for every
method at every SNR, annotating the test-set size on the figure title — so the stability of small gaps
between methods (e.g. greedy-split vs MANET-GNN for unicast) is explicit.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/AlterTomer/Decentralized-MANET.git
cd Decentralized-MANET
```

### Create Environment

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Main Dependencies

Typical dependencies include:

* PyTorch
* PyTorch Geometric
* NumPy
* SciPy
* Matplotlib
* NetworkX
* QuaDRiGa (optional, for geometry-based channels)

---

## Running Experiments

Experiments are driven by `.ini` config files: a **training** config per model and a **benchmark**
(comparison) config per evaluation. The exact configs of every result in the paper are in
[config/paper/](config/paper), and [docs/REPRODUCE.md](docs/REPRODUCE.md) gives the full recipe. Run
all scripts from the repository root with `PYTHONPATH=.`.

```bash
python scripts/install_paper_configs.py <data root>      # install config/paper/ into a data folder
set MANET_DATA_ROOT=<data root>                         # PowerShell: $env:MANET_DATA_ROOT = "<data root>"
```

### Train MANET-GNN

Example training config (many-to-many, full CSI):

```ini
[Train Parameters]
seed: 7241
mode: multiunicast
B: 6
L: 3                      # gated layers (= 2L communication rounds)
n: 10
tx: 0, 2, 4, 6
rx: 3, 5, 7, 9
sigma: 1
SNR: 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
dropout: 0.2
lr: 5e-4
wd: 3e-5
noise conditioning: 1     # the SNR is an input feature
decentralized training: 1
decentralized validation: 1
z mode: candidate         # candidate-route routing head (F3-F5)
routing tau start: 1.0
routing tau end: 0.05
reduce: fair
val reduce: fair
max route hops: 3
grad clip: 1
epochs: 100
num samples: 2000
grad batch: 8
mono: 0.1
LMMSE estimation: 0       # 1 = train on LMMSE-estimated CSI (loss on the true channel)

[Files]
channel path: <DATA_ROOT>\Quadriga channels\Train_H_seed_48312_n_10_B_6_samples_2000.mat
ckpt dir: <DATA_ROOT>\Trained Models\multiunicast
figs dir: <DATA_ROOT>\Figs\multiunicast
prefix: Quadriga_Multiunicast_n_10_B_6_K_4_z_candidate_noisecond
```

```bash
python scripts/Decentralized_MANET_Simulation.py <training config(s)>
python MANET_FFN/main.py <FFN config(s)>                 # FFN baseline
```

### Benchmark MANET-GNN

```bash
python scripts/Optimizer_vs_GNN.py --configs <benchmark config(s)>
python scripts/run_benchmarks_parallel.py --workers 4   # all benchmark configs, in parallel
```

A benchmark config points to the test set, the trained GNN and FFN, and the outputs, and sets the SNR
grids (`snr list` for all methods, `optimizer snr` for the centralized optimizer); see
[config/paper/](config/paper) and [docs/DESIGN.md §8](docs/DESIGN.md#8-config-key-reference).

---

## Experimental Results

The paper evaluates the five frameworks (unicast, multicast, multicommodity, convergecast,
many-to-many) on QuaDRiGa frequency-selective channels, under full and estimated CSI, against the
benchmarks above. It further studies:

* generalization to sparser and denser ER graphs (p = 0.1–0.5) and to non-ER topologies (random
  geometric and hub/periphery graphs), without retraining;
* heterogeneous radios, where each node supports only a subset of the channels;
* the rate achieved versus the number of communication rounds;
* soft (training) versus hard (inference) route selection;
* the effect of channel-estimation errors.

MANET-GNN consistently approaches centralized optimization performance while preserving decentralized
operation. Results are reported in Section IV of the paper.

---

## Paper

The current preprint is available here:

[MANET-GNN TCOM Preprint](./paper/MANET-GNN%20Learned%20Decentralized%20Optimization%20of%20Power%20Allocation%20in%20Multi-Channel%20MANETs.pdf)

> Under review in IEEE Transactions on Communications.

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{alter2026manetgnn,
  title={MANET-GNN: Learned Decentralized Optimization of Power Allocation in Multi-Channel MANETs},
  author={Alter, Tomer and Shlezinger, Nir and Segal, Michael},
  journal={IEEE Transactions on Communications},
  year={2026}
  note={Under review}
}
```

---

## Related Publication

```text
Extended journal version under review.
T. Alter, N. Shlezinger, and M. Segal,
"MANET-GNN: Learned Decentralized Optimization of Power Allocation in Multi-Channel MANETs,"
IEEE Transactions on Communications, 2026.
```

Conference predecessor:

```text
T. Alter, N. Shlezinger, and M. Segal,
"Decentralized Multi-Channel MANET Power Optimization Using Graph Neural Networks,"
IEEE ICC 2026.
```

---

## Future Directions

Planned extensions include:

* Decentralized OFDMA optimization
* Joint routing and scheduling
* Continual learning over dynamic topologies
* Hypernetwork-based multi-framework adaptation
* Interference-aware channel allocation
* Temporal graph learning

---

## License

Specify your repository license here.

Example:

```text
MIT License
```

---

## Contact

**Tomer Alter**
Ben-Gurion University of the Negev
Email: [tomeralt@post.bgu.ac.il](mailto:tomeralt@post.bgu.ac.il)

GitHub: [https://github.com/AlterTomer](https://github.com/AlterTomer)
