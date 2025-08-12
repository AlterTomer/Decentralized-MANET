# Decentralized MANET Power Allocation with GNNs
This repository implements and evaluates decentralized power allocation algorithms for **multi-band OFDM Mobile Ad-Hoc Networks (MANETs)**.
It compares **Graph Neural Network (GNN)-based approaches** against centralized optimization and analytical lower bounds under various SNR conditions.
Features
**Custom Graph Dataset Generation**
Generates connected graphs with adjustable parameters (number of nodes, frequency bands, noise variance).
Supports loading precomputed channels from .mat files.

**Model Architectures**
Gated GCN and GAT-based GNN variants for decentralized power allocation.

**Centralized Optimization**
Reference ADAM-based optimizer for benchmarking.

**Lower Bound Calculation**
Analytical bound based on the strongest link among all weakest links per band.

**SNR Sweep Evaluation**
Automatic comparison of GNN, centralized optimization, and lower bounds across a range of SNRs.

**Training Tools**
Cosine warm restarts the scheduler.
Optional Stochastic Weight Averaging (SWA).
Best checkpoint saving with old checkpoint cleanup.

**Project Structure**
.
├── data/                    # (Optional) Precomputed datasets and channels
├── models/                  # GNN architectures
├── utils/                 
│   ├── DataUtils.py              # Graph generation & dataset building
│   ├── PathUtils.py              # Path finding & tensor conversion
│   ├── MetricUtils.py            # Rate calculation & loss functions
│   ├── CentralizedUtils.py       # Centralized optimization & lower bounds
│   ├── TrainUtils.py             # Schedulers, checkpoint saving
|   ├── ComparisonUtils.py        # Rate optimization comparison 
|   ├── ConfigUtils.py            # Config files handling
├── training_scripts/        # Training entry points & configs
├── evaluation/              # Scripts for SNR sweep & performance plots
├── config/                  # INI/YAML config files
└── README.md

**Installation**
git clone https://github.com/AlterTomer/Decentralized-MANET.git
cd Decentralized-MANET
pip install -r requirements.txt

### CUDA Support
This project supports GPU acceleration via CUDA for faster training and evaluation.  
If you have a CUDA-enabled GPU and compatible drivers, install the CUDA-enabled version of PyTorch by following the instructions at [PyTorch.org](https://pytorch.org/get-started/locally/).
If CUDA is not available, the code will automatically run on CPU (with slower performance).

**Usage**
**1. Train a GNN**
python training_scripts/Decentralized_MANET_Simulation.py --config config/train.ini

**2. Evaluate at Multiple SNRs**
python evaluation/Optimizer_vs_GNN.py --config config/eval.ini
