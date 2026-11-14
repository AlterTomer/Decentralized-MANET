# Reproducing the Paper's Results

This guide walks through the full pipeline behind the results of the revised manuscript: channel
generation, training of MANET-GNN and the FFN baseline, the benchmark sweeps, and the additional
studies (graph density, non-ER topologies, heterogeneous radios, soft vs. hard routing, rate vs.
communication rounds, the CSI-estimation analysis, size generalization, and depth ablation). Every config used for the paper is shipped in
[`config/paper/`](../config/paper).

> The heavy steps are the centralized-optimizer benchmark sweeps (tens of GPU-hours per main config;
> see §4.5) and the trainings (a few hours each on a single GPU). Everything else takes minutes.

---

## 0. Setup

### 0.1 Python

```bash
git clone https://github.com/AlterTomer/Decentralized-MANET.git
cd Decentralized-MANET
python -m venv venv && venv\Scripts\activate        # Windows (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

All scripts are run **from the repository root with `PYTHONPATH=.`** (PowerShell: `$env:PYTHONPATH = "."`).

### 0.2 MATLAB + QuaDRiGa (channel generation only)

The channels are generated with [QuaDRiGa](https://quadriga-channel-model.de/) in MATLAB
(`matlab/`). Add QuaDRiGa's `quadriga_src` folder and this repo's `matlab/` folder to the MATLAB path
(each script has an `addpath` block and an `OUTPUT_DIR` at the top — point `OUTPUT_DIR` to
`<DATA_ROOT>\Quadriga channels`).

### 0.3 Data root and configs

All inputs and outputs live under one **data root** with this layout:

```text
<DATA_ROOT>/
├── Config Files/          # the configs (installed from config/paper/)
├── Quadriga channels/     # generated channel sets (.mat / .pt)
├── Trained Models/        # GNN checkpoints (per framework) and FFN/ (per framework)
├── Figs/                  # benchmark figures (.png)
└── Figures Data/          # benchmark results (.pkl, .csv)
```

Install the configs (this replaces the `<DATA_ROOT>` placeholder and creates the output folders), then
point the scripts at the data root with the `MANET_DATA_ROOT` environment variable:

```bash
python scripts/install_paper_configs.py D:\manet_data
set MANET_DATA_ROOT=D:\manet_data                  # cmd;  PowerShell: $env:MANET_DATA_ROOT = "D:\manet_data"
```

The config tree mirrors the framework folders used by the scripts:

| Framework | Paper | Code `mode` | Config folder | File tag |
|-----------|-------|-------------|---------------|----------|
| Unicast | F1 | `single` | `Single Tx-Rx/` | `unicast` |
| Multicast | F2 | `multicast` | `Multicast/` | `multicast` |
| Multicommodity | F3 | `multi` | `Multicommodity/` | `multicommodity` |
| Convergecast | F4 | `converge` | `Convergencecast/` | `convergecast` |
| Many-to-Many | F5 | `multiunicast` | `Multiunicast/` | `multiunicast` |

Config names: `<Folder>/R2/ChainedGNN_<tag>_<CSI>_R2.ini` (GNN training),
`FFN/<Folder>/FFN_<tag>_<CSI>_R2.ini` (FFN training), and `<Folder>/R2/comp_<tag>_<study>_R2.ini`
(benchmarks), with `<CSI>` = `FullCSI` or `EstimatedCSI`.

---

## 1. Channel and topology generation

All networks are generated with QuaDRiGa, scenario `5G-ALLSTAR_DenseUrban`, carrier 2.4 GHz, `B = 6`
frequency bins over 10 MHz, and per-link normalization (each link's `B` coefficients are normalized to
zero mean and unit standard deviation). Nodes lie in a 100 × 100 m urban block (x, y uniform; 20 % of
the nodes at 3 m height, the rest at 1.5 m); one layout is shared by all networks of a set, except for
the random geometric graphs, which draw a new layout per network. Every test network is **connected**, and the test sets are
drawn from random-number streams that are disjoint from those of the training set.

| Set | Script | Output file (in `Quadriga channels/`) | Used by |
|-----|--------|---------------------------------------|---------|
| Training, \|V\| = 10, ER p = 0.3, 2000 networks | `matlab/quadriga_demo.m` (`n = 10`) | `Train_H_seed_48312_n_10_B_6_samples_2000.mat` | F1, F3–F5 training |
| Training, \|V\| = 6, ER p = 0.3, 2000 networks | `matlab/quadriga_demo.m` (`n = 6`) | `Train_H_seed_48312_n_6_B_6_samples_2000.mat` | F2 training |
| Main test, \|V\| = 10, p = 0.3, 500 networks | `matlab/generate_R2_test_sets.m` | `Test_H_seed_90210_p_0p3_n_10_B_6_samples_500.mat` | main benchmarks F1, F3–F5 |
| Main test, \|V\| = 6, p = 0.3, 500 networks | `matlab/generate_R2_test_sets.m` | `Test_H_seed_90210_p_0p3_n_6_B_6_samples_500.mat` | main benchmarks F2 |
| Density test, 100 networks per p ∈ {0.1,…,0.5} | `matlab/generate_R2_test_sets.m` | `Test_H_seed_31337_p_0p1_0p2_0p3_0p4_0p5_n_10_B_6_samples_500.mat` | density study |
| Random geometric (unit-disk, R = 42.5 m), 200 networks | `matlab/generate_R2_topology_sets.m` | `Test_H_RGG_seed_520000_n_10_B_6_samples_200.mat` | topology study |
| Hub/periphery (Chung–Lu), 200 networks | `matlab/generate_R2_topology_sets.m` | `Test_H_HUB_seed_610000_n_10_B_6_samples_200.mat` | topology study |
| Heterogeneous radios, test (from the main test set) | `python scripts/make_heterogeneous_radio_set.py` | `Test_H_HETRADIO_seed_90210_p_0p3_n_10_B_6_samples_500.pt` | heterogeneous-radio study |
| Heterogeneous radios, training (from the training set) | `python scripts/make_heterogeneous_radio_train_set.py` | `Train_H_HETRADIO_seed_48312_n_10_B_6_samples_2000.pt` | heterogeneous-radio retraining |

* `generate_R2_test_sets.m` produces the three test sets of its `RUNS` table in one run.
* Heterogeneous radios: every node supports each channel independently with probability 2/3 (at least
  one channel per node); an unsupported channel is represented by zero coefficients on all the node's
  links, and a link without a common channel is removed. Support patterns are redrawn until the network
  stays connected (training set: until its connected components are unchanged).
* The two `.pt` builders read `$MANET_DATA_ROOT\Quadriga channels`, so run them after the MATLAB sets.
* 30-node sets (size generalization and depth ablation, §4.4): `matlab/quadriga_demo.m` with `n = 30` for the
  training set, and a row `"Test", 30, <p>, 200, <seed>` in the `RUNS` table of `generate_R2_test_sets.m` for
  the 200 test networks. Name the files as in `config/paper/Generalization/` (or edit the configs).

---

## 2. Training

### 2.1 MANET-GNN

```bash
python scripts/Decentralized_MANET_Simulation.py "%MANET_DATA_ROOT%\Config Files\Multicommodity\R2\ChainedGNN_multicommodity_FullCSI_R2.ini"
```

The script accepts any number of configs on the command line (run without arguments, it trains the two
convergecast configs). Train the **ten** models of the paper: the five frameworks × {`FullCSI`,
`EstimatedCSI`}. A checkpoint is saved in the config's `ckpt dir` as
`<prefix>_L<layers>_val<best validation>_<timestamp>.pth`.

Settings shared by all frameworks: 2000 training networks (1600 train / 400 validation), trained over
the SNRs {0, 5, …, 50} dB, 100 epochs, AdamW (weight decay 3e-5), cosine schedule, dropout 0.2, gradient batch 8,
gradient clipping 1, monotonicity weight `mono = 0.1`, `noise conditioning = 1` (the SNR is an input
feature). Framework-specific settings:

| | F1 Unicast | F2 Multicast | F3 / F4 / F5 |
|---|---|---|---|
| \|V\|, Tx → Rx | 10, 4 → 9 | 6, 0 → {2, 4} | 10; F3: 4 → {1, 6, 7, 9}; F4: {0, 2, 4, 6} → 5; F5: {0, 2, 4, 6} → {3, 5, 7, 9} |
| gated layers `L` (rounds = 2L) | 2 | 2 | 3 |
| learning rate | 1e-3 | 5e-4 | 5e-4 |
| routing | per-edge, sparsity 0.05 | per-edge, sparsity 0.2 | candidate routes (`z mode = candidate`), ≤ 3 hops, τ_r annealed 1 → 0.05, `reduce = fair` |
| other | `p head bias init = 1.5` | — | decentralized training/validation |

**Estimated CSI** (`LMMSE estimation = 1`): the model's input is the LMMSE channel estimate, while the
training loss is evaluated on the true channel (§6 of [DESIGN.md](DESIGN.md)).

### 2.2 FFN baseline (B7)

```bash
python MANET_FFN/main.py "%MANET_DATA_ROOT%\Config Files\FFN\Multicommodity\FFN_multicommodity_FullCSI_R2.ini"
```

Same training data, SNRs, and CSI setting as the matching GNN; 4 fully connected layers of width 512,
lr 1e-3, 100 epochs, 20 % validation. The FFN does not receive the Tx/Rx roles, so its training Tx/Rx
**must equal** the benchmark's (the shipped configs already match). It is saved as the folder
`<ckpt dir>\<prefix>_ffn`.

### 2.3 Point the benchmark configs at the trained models

Every benchmark config has a `model path` (GNN checkpoint) and an `ffn path` (FFN folder). The shipped
values carry our checkpoint names; replace them with yours:

* `comp_<tag>_EstimatedCSI_R2.ini` and `comp_<tag>_EstModelTrueCSI_R2.ini` → the **EstimatedCSI** models;
* every other `comp_<tag>_*_R2.ini` (FullCSI, density, rgg, hub, hetradio) → the **FullCSI** models;
* `comp_<tag>_FullCSI_hetradio_retrained_R2.ini` → the heterogeneous-radio models of §2.4.

For convergecast, `python scripts/fill_convergecast_paths.py` does this automatically (newest
checkpoint of each prefix).

### 2.4 Heterogeneous-radio retraining (F3–F5)

```bash
python scripts/run_hetradio_training.py          # GNN then FFN, ChainedGNN_<tag>_hetradio_R2.ini / FFN_<tag>_hetradio_R2.ini
python scripts/fill_hetradio_paths.py            # writes the checkpoints into comp_<tag>_FullCSI_hetradio_retrained_R2.ini
```

These configs equal the FullCSI training configs except for the training set
(`Train_H_HETRADIO_...pt`) and the checkpoint prefix (`scripts/make_hetradio_retrain_configs.py`
generated them). `fill_hetradio_paths.py` pins the checkpoints used in the paper by name and otherwise
takes the newest one. At inference, the routing head excludes candidate routes with a hop that does not
support the band (`CandidateRoutingHead.mask_dead_routes`, on by default; an exact no-op on homogeneous
networks).

---

## 3. Benchmarks

### 3.1 Running

```bash
python scripts/run_benchmarks_parallel.py --workers 4 --dry-run   # list configs + cost estimate
python scripts/run_benchmarks_parallel.py --workers 4             # run all comp_*_R2.ini configs
python scripts/run_benchmarks_parallel.py --workers 4 --only density,rgg
python scripts/Optimizer_vs_GNN.py --configs <cfg1.ini> <cfg2.ini>  # single process
```

The runner collects `Config Files/*/R2/comp_*_R2.ini`, skips configs whose channel file or model does
not exist yet, balances the rest over worker processes by estimated cost, and logs to
`logs/benchmark_worker_<k>.log`.

Each config evaluates all eight methods on its test set and writes the figure (`fig path`) and the
results pickle (`fig data path`). Benchmark-specific keys:

| Key | Meaning | Main configs | Side studies |
|-----|---------|--------------|--------------|
| `num samples` | test networks used (the first N of the file) | 500 | 500 (density, hetradio), 200 (rgg, hub) |
| `snr list` | SNRs (dB) of all methods | 0, 5, …, 50 | 0, 5, …, 50 |
| `optimizer snr` | SNRs of the centralized optimizer (`999` = skip it) | 0, 5, …, 50 | 10, 20, 30 |
| `LMMSE estimation` | 1 = methods decide from the LMMSE estimate | 0 / 1 | 0 |
| `decentralized gnn inference`, `hard candidate gnn inference`, `max route hops` | F3–F5: deployed MANET-GNN, i.e., local candidate routes of ≤ 3 hops and hard per-band argmax | 1, 1, 3 | same |

Scoring rules (identical for every method):

* the rate is the original max–min objective (hard min/max), with co-channel interference;
* **estimated CSI**: every method decides from the estimate and is **scored on the true channel**; the
  SNR → noise mapping uses the true-channel power, so full- and estimated-CSI runs share the noise level;
* **heterogeneous radios**: before scoring, each node drops power on channels it does not support and
  rescales the rest to the same total (a local operation; a no-op on homogeneous networks);
* the centralized optimizer (B1) runs AdamW for 1000 iterations with step size 0.1 from the Greedy-Split,
  Widest-Path, and MANET-GNN allocations and keeps the best iterate, so it is never below any of them.

### 3.2 Method keys in the results pickle

| Paper | Results key |
|-------|-------------|
| MANET-GNN | `gnn` |
| B1 Centralized Optimizer | `centralized` |
| B2 Equal-Split | `equal power` |
| B3 Centralized Greedy-Split (CGS) | `greedy maxlink` |
| B4 Decentralized Greedy-Split (DGS) | `greedy maxlink decentralized` |
| B5 Centralized Widest-Path (CWP) | `strongest bottleneck` |
| B6 Decentralized Widest-Path (DWP) | `strongest bottleneck decentralized` |
| B7 FFN | `ffn` |

`results[key][snr]` is the mean rate, `results["sem"][key][snr]` its standard error (the figures show
±1.96·SEM), `results["per_sample"][key][snr]` the per-network rates (dataset order), `results["n_test"]`
the number of networks, and, for files that store it, `results["p_vec"]` the ER edge probability of every
network (density study).

### 3.3 Which config produces which result

| Paper result | Configs (`comp_<tag>_..._R2.ini`) | How the reported numbers are obtained |
|--------------|-----------------------------------|----------------------------------------|
| Benchmark figures, full CSI (F1–F5) | `FullCSI` | figure as written by the benchmark |
| Benchmark figures, estimated CSI (F1–F5) | `EstimatedCSI` | figure as written by the benchmark |
| Topology table (ER / hub / geometric) | `FullCSI`, `FullCSI_hub`, `FullCSI_rgg` (F1, F3–F5) | means at 10/20/30 dB |
| Density table | `FullCSI_density` (F1, F3–F5) | means per `p_vec` value at 10/20/30 dB; "within reach" = networks in which every destination is within L/2 hops of its source |
| Heterogeneous-radio table | `FullCSI_hetradio` (zero-shot) and `FullCSI_hetradio_retrained` (F3–F5) | means at 10/20/30 dB, all networks and within reach (single-channel route of ≤ L/2 hops) |
| CSI table (estimated / full, same model) | `EstimatedCSI` and `EstModelTrueCSI` | `python scripts/same_model_csi_ratio.py` (§4.3) |
| Size generalization and depth ablation figure (F5) | `Generalization/size_generalization.ini`, `Generalization/depth_ablation.ini` (study configs) | `scripts/SizeGeneralization.py`, `scripts/DepthAblation.py` (§4.4) |

---

## 4. Additional studies

### 4.1 Soft vs. hard route selection

```bash
python scripts/RoutingSoftVsHard.py
```

For F3–F5 (the `EstimatedCSI` benchmark configs), evaluates the same checkpoint with hard per-band
argmax routing (deployed), per-band softmax at the final training temperature (τ_r = 0.05), and at
τ_r = 1, and records the mean maximal route probability. Output: `<fig data stem>_soft_vs_hard.csv`.

### 4.2 Rate vs. number of communication rounds

```bash
python scripts/RateVsCommunication.py
```

For F1, F3, F4, F5 (`FullCSI` configs) at 10/20/30 dB: MANET-GNN decoded after each gated layer (2, 4,
…, L rounds), the decentralized Greedy-Split and Widest-Path benchmarks with Bellman–Ford stopped after
t = 1, …, |V| − 1 rounds, Equal-Split (0 rounds), and the optimizer as a reference line (read from the
`FullCSI` benchmark pickle, so run §3 first). Output: `<fig data stem>_rate_vs_comm.{csv,pkl,png}`.

### 4.3 CSI estimation with the same model

The full- and estimated-CSI curves come from separately trained models. To isolate the effect of the
estimation errors, the estimated-CSI models are also run with the true channels as input
(`comp_<tag>_EstModelTrueCSI_R2.ini`, shipped; `scripts/make_same_model_csi_configs.py` generates them
from the `EstimatedCSI` configs; they skip the optimizer):

```bash
python scripts/run_benchmarks_parallel.py --workers 5 --only EstModelTrueCSI
python scripts/same_model_csi_ratio.py       # per framework: estimated-input / true-input rate, with paired 95% CIs
```

### 4.4 Size generalization and depth ablation (many-to-many)

Both studies evaluate **MANET-GNN only** (no benchmarks), with the same inference and scoring as the
benchmark (decentralized candidate routing, hard per-band argmax, rate on the true channel). A *study
config* lists any number of trained models (`[Model ...]`, each with its own depth) and test sets
(`[Test ...]`, each with its own size, Tx/Rx nodes, and channel file); every model is evaluated on every
test set. Format: `utils/GNNStudyUtils.py`; examples: `config/paper/Generalization/`.

**Training the 30-node models.** `ChainedGNN_multiunicast_n30.ini` is the many-to-many full-CSI training
config with |V| = 30 and 5 gated layers (L = 10). The depth-ablation models are copies that differ only in
depth (`ChainedGNN_multiunicast_n30_L{1,2,3,7}.ini`, i.e., L = 2, 4, 6, 14), written by

```bash
python scripts/DepthAblation.py --make-train-configs "%MANET_DATA_ROOT%\Config Files\Generalization\ChainedGNN_multiunicast_n30.ini" --layers 1 2 3 7
python scripts/Decentralized_MANET_Simulation.py <each of the five configs>
```

(`--make-train-configs` sets `L` and `max route hops` to each depth and keeps everything else.)

**Evaluating.** Put the trained checkpoints into the `path` lines of the study configs, then

```bash
python scripts/DepthAblation.py "%MANET_DATA_ROOT%\Config Files\Generalization\depth_ablation.ini"
python scripts/SizeGeneralization.py "%MANET_DATA_ROOT%\Config Files\Generalization\size_generalization.ini"
```

* `depth_ablation.ini`: the five depths on the 200 test networks with |V| = 30 → mean rate vs SNR per depth
  (`<out prefix>_<test>.png`, the paper's depth-ablation panel) and mean rate vs L at the `summary snr`
  values (`..._vs_depth.png`).
* `size_generalization.ini`: the |V| = 10 model of the main results (L = 6) and the |V| = 30 model (L = 10)
  on the same 30-node networks → mean rate vs SNR per model (the paper's size-generalization panel). Add
  `[Test ...]` sections for more sizes; with two or more, `..._vs_size.png` plots the rate vs |V|. A model
  applies to any size as long as the number of messages K matches its training.
* Outputs (in `out dir`): `<out prefix>.pkl` (per-network rates, means, 95% CIs per model and test set),
  `.csv`, and the figures. Models or test files that do not exist yet are skipped.
* The checkpoint is checked against the model: a depth or `noise conditioning` mismatch raises an error
  instead of silently loading a partial model.

### 4.5 Compute budget

The centralized optimizer dominates the cost. One optimizer run (1000 iterations, one initialization)
takes about 3 s (F1/F2) to 8 s (F3–F5) per network and SNR on a single GPU, i.e., 500 networks × 11
SNRs × 3 initializations ≈ 12–37 GPU-hours per main config. `run_benchmarks_parallel.py --dry-run`
prints the estimate for each config.

---

## 5. Other utilities

* `scripts/diagnose_offroute.py` — prints, per method, how concentrated the allocation is on its
  route(s) and how interference-limited the route is (no files written).
* `scripts/install_paper_configs.py` — installs `config/paper/` into a data root (§0.3).
* `utils/GNNStudyUtils.py` — the MANET-GNN-only evaluation engine of §4.4 (reusable for other
  model × test-set studies).
