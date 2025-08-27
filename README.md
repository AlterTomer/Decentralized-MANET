# Decentralized MANET Power Allocation with GNNs

This repository implements and evaluates **decentralized power allocation** algorithms for  
**multi-band OFDM Mobile Ad-Hoc Networks (MANETs)** using **Graph Neural Networks (GNNs)**.  
We compare our GNN approach against a **centralized AdamW optimizer** and a **waterfilling-based benchmark** under Rayleigh and QuaDRiGa channel models.

---

## Features

- **Custom Graph Dataset Generation**  
  - Generates connected graphs with adjustable parameters: number of nodes, frequency bands, noise variance.  
  - Supports loading precomputed channels from `.mat` files.

- **Model Architectures**  
  - GatedGCN + ChainedGNN architectures for decentralized power allocation.  
  - Edge-conditioned **FiLM** modulation and **Jumping Knowledge (JK)** aggregation.

- **Centralized Optimization**  
  - Reference **AdamW** optimizer for global (oracle) baseline.

- **Waterfilling Benchmark**  
  - Analytical single-path power allocation via waterfilling.

- **SNR Sweep Evaluation**  
  - Automated comparison of **GNN**, **centralized AdamW**, and **waterfilling** across SNR values.

- **Training Tools**  
  - Cosine warm restarts scheduler.  
  - Optional SWA.  
  - Best checkpoint saving with automatic cleanup.

---

## Project Structure

```text
├── README.md
├── config/                           # INI/YAML config files
├── data/                             # (Optional) precomputed datasets & channels
├── datasets/
│   └── GraphDataSet.py               # PyG Dataset & collate helpers
├── models/
│   ├── models.py                     # GatedGCNLayer, ChainedGNN
│   └── GraphNetAux.py                # Training loops + τ scheduling
├── scripts/
│   ├── Decentralized_MANET_Simulation.py   # End-to-end training & plots
│   └── Optimizer_vs_GNN.py                 # AdamW vs GNN vs waterfilling eval
├── utils/
│   ├── PathUtils.py                  # Path search + tensorization
│   ├── TensorUtils.py                # Power init & normalization
│   ├── TrainUtils.py                 # LR schedulers, checkpoint helpers
│   ├── CentralizedUtils.py           # Centralized AdamW + waterfilling
│   ├── ComparisonUtils.py            # SNR sweep comparison harness
│   ├── ConfigUtils.py                # CLI / INI parsing utilities
│   ├── EstimationUtils.py            # LMMSE estimation helpers
│   ├── FilesUtils.py                 # Atomic save + checkpoint management
│   └── MetricUtils.py                # Rate computation & smooth-min approximation
└── visualization/
    └── GraphingAux.py                # Training curves & SNR plots


---
 **Installation** 
git clone https://github.com/AlterTomer/Decentralized-MANET.git
cd Decentralized-MANET
pip install -r requirements.txt

---
**CUDA Support**
This project supports GPU acceleration via CUDA for faster training and evaluation.
If you have a CUDA-enabled GPU and compatible drivers, install the CUDA-enabled version of PyTorch at
https://pytorch.org/get-started/locally/
If CUDA is not available, the code automatically falls back to CPU.

**Usage**
1) Train a GNN: python scripts/Decentralized_MANET_Simulation.py --config config/train.ini
2) Evaluate Performance at Multiple SNRs: python scripts/Optimizer_vs_GNN.py --config config/eval.ini

**Channel Generation (Optional)**

For realistic, geometry-based channels, this repository supports datasets generated with QuaDRiGa (Fraunhofer HHI):
https://github.com/fraunhoferhhi/QuaDRiGa

To generate your own channels:

Download the official QuaDRiGa MATLAB package and place it under matlab/ (local, not required in this repo).

Use your MATLAB scripts (e.g., quadriga_demo.m) to export channel datasets as .mat files.

Important: this project expects the Fourier transform of the channel H(f), i.e., the frequency-domain response, not the raw time-domain impulse response.

If you skip this step, the repository falls back to Rayleigh fading for training and evaluation.

**Results**

Our ChainedGNN achieves approximately 80% of centralized AdamW performance and about 75% of the waterfilling benchmark, while requiring significantly fewer computations and producing predictions in a single forward pass.

Method comparison (n nodes, B frequency bands, L GatedGCN layers, I centralized iterations, P number of candidate paths between tx and rx):
Example for n = 8, B = 6, L = 3, I = 50, P = 20 
ChainedGNN (ours):
• Global CSI: No
• Computation per sample: O(L·n²(1 + B) ≈ 530 operations
• Scalability: High

Centralized AdamW:
• Global CSI: Yes
• Computation per sample: O(I·B·n²) ≈ 19,200 operations
• Scalability: Low

Waterfilling:
• Global CSI: Yes
• Computation per sample: O(P·B·n²) ≈ 7,680 operations
• Scalability: Low

Example figures (update paths if needed):

Training/validation (Rayleigh):
Train Quadriga_Lin_TAU_ChainedGNN_3 layers 6 bands network.png

Comparison (Rayleigh):
Comparison lin tau.png

Comparison (QuaDRiGa):
Comparison lin tau Quadriga.png
