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

* Rayleigh fading channels
* QuaDRiGa frequency-selective channels
* Full CSI and noisy CSI regimes

and demonstrated near-centralized performance across all communication frameworks.

---

## Key Features

### Unified Multi-Framework MANET Optimization

A single formulation supporting:

| Framework | Description    |
| --------- | -------------- |
| F1        | Unicast        |
| F2        | Multicast      |
| F3        | Multicommodity |
| F4        | Convergecast   |

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

> Adjust paths below if you later reorganize the repository.

```text
Decentralized-MANET/
│
├── Multicast/                 # Multicast subgraph and objective utilities
├── Multicommodity/            # Multicommodity objective utilities
├── configs/                   # Experiment configurations
├── datasets/                  # Dataset generation and loading utilities
├── matlab/                    # QuaDRiGa-related utilities and channel generation
├── models/                    # MANET-GNN architectures and layers, aux function for training and evaluation
├── scripts/                   # Training and benchmarking scripts
├── utils/                     # Graph, routing, normalization, and helper functions
├── visualization/             # Visualization and graphing utilities
├── requirements.txt
└── README.md
```

---

## System Model

We consider a dynamic multi-hop multi-channel MANET represented by an undirected graph:

```math
G = (V, E)
```

where:

* Nodes represent mobile devices
* Edges represent reciprocal wireless links
* Each link contains (B) orthogonal frequency channels

The achievable rate over link ((i,j)) and channel (b) is:

```math
R_{i \rightarrow j}^{(b)} = \log_2\left(1 + \frac{|h_{i\rightarrow j}^{(b)}|^2 p_{i\rightarrow j}^{(b)2}}{\sigma_b^2}\right)
```

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

The design follows the architecture described in Sections III-A and III-B of the paper. fileciteturn0file0L330-L365

---

## Training

MANET-GNN is trained using unsupervised learning.

The training objective directly maximizes the end-to-end communication rate:

```math
\max_P \min_k R_k^{E2E}(P)
```

The total loss combines:

* Rate maximization
* Monotonic improvement regularization

Training setup from the paper:

* AdamW optimizer
* Cosine learning-rate scheduling
* Dropout regularization
* Rayleigh and QuaDRiGa channels
* SNR range: 0–20 dB

---

## Benchmarks

The repository includes comparisons against:

| Baseline              | Description                             |
| --------------------- | --------------------------------------- |
| Centralized Optimizer | AdamW optimization over the full MANET  |
| Equal-Split           | Uniform power allocation                |
| Greedy-Split          | Shortest-path-based heuristic           |
| Best Single Channel   | Bottleneck-based single-band allocation |

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

Experiments are launched through the simulation/training script using an `.ini` configuration file.

The training script reads all experiment settings from the config file, including:

- Communication framework
- Number of nodes and bands
- SNR values
- Optimizer settings
- Checkpoint and figure directories
- CSI estimation mode
  
### Train MANET-GNN
## Configure an Experiment

Create or edit an `.ini` file. Example:

```ini
[Train Parameters]
seed: 7241
mode: multi
B: 6
L: 2
n: 10
tx: 4
rx: 1, 6, 7, 9
SNR: 0, 4, 8, 12, 16, 20
dropout: 0.2
lr: 1e-3
wd: 3e-5
epochs: 100
num samples: 2000
grad batch: 8
mono: 0.5
LMMSE estimation: 0

[Files]
channel path: path/to/quadrigat_channels.mat  # If you don't want to use generated Rayleigh channels
ckpt dir: path/to/checkpoints
figs dir: path/to/figures
prefix: _Rayleigh_Multicommodity_n_10_B_6_K_4_L_2
```


```bash
python Decentralized_MANET_Simulation.py --config path/to/config.ini
```

### Benchmark MANET-GNN

Benchmarking is performed using `Optimizer_vs_GNN.py` and a comparison `.ini` file.

Example config:

```ini
[Train Parameters]
seed: 15337
mode: multi
B: 6
L: 3
n: 12
tx: 7
rx: 1, 4, 5, 8
sigma: 1
dropout: 0.2
num samples: 20
LMMSE estimation: 0

[Files]
channel path: path/to/quadrigat_channels.mat  # If you don't want to use generated Rayleigh channels
model path: path/to/trained_model.pth
fig path: path/to/save/benchmark_figure.png
fig data path: path/to/save/benchmark_results.pkl

```bash
python Optimizer_vs_GNN.py --config path/to/config.ini
```
---

## Experimental Results

The paper evaluates:

* Unicast
* Multicast
* Multicommodity
* Convergecast

under:

* Rayleigh fading
* QuaDRiGa channels
* Full CSI
* Noisy CSI

MANET-GNN consistently approaches centralized optimization performance while preserving decentralized operation.

Representative results are shown in Section IV of the paper. fileciteturn0file0L705-L739

---

## Paper

The current preprint is available here:

[MANET-GNN TCOM Preprint](paper/MANET-GNN Learned Decentralized Optimization of Power Allocation in Multi-Channel MANETs.pdf)

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
