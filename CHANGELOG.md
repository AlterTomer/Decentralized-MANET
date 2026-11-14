# Changelog

## Second revision (November 2026)

Work accompanying the second revision of the paper. The full reproduction recipe is in
[docs/REPRODUCE.md](docs/REPRODUCE.md); the configs of every reported result are in
[config/paper/](config/paper).

### New studies

| Study | What | Why | Code |
|-------|------|-----|------|
| Graph density | Zero-shot evaluation of the p = 0.3 models on ER graphs with p ∈ {0.1, …, 0.5} (100 networks each), at 10/20/30 dB, also restricted to networks whose destinations are within the GNN's L/2-hop reach | How MANET-GNN behaves on sparser/denser networks than it was trained on | `matlab/generate_R2_test_sets.m`, `comp_*_FullCSI_density_R2.ini` |
| Non-ER topologies | Random geometric (unit-disk) graphs and hub/periphery (Chung–Lu) graphs, 200 networks each | Generalization beyond the ER training distribution | `matlab/generate_R2_topology_sets.m`, `comp_*_FullCSI_{rgg,hub}_R2.ini` |
| Heterogeneous radios | Each node supports a random subset of the channels (each w.p. 2/3); zero-shot and retrained models (F3–F5) | Nodes with different radio capabilities, as allowed by the system model | `scripts/make_heterogeneous_radio_{set,train_set}.py`, `scripts/run_hetradio_training.py`, `comp_*_hetradio*_R2.ini` |
| Soft vs. hard routing | Same checkpoint with hard per-band argmax vs. softmax at τ_r = 0.05 and 1 | Quantify the gap between the training relaxation and deployment | `scripts/RoutingSoftVsHard.py` |
| Rate vs. communication rounds | MANET-GNN per layer vs. Bellman–Ford benchmarks stopped after t rounds | Compare decentralized methods at an equal signaling budget | `scripts/RateVsCommunication.py` |
| Same-model CSI analysis | Estimated-CSI models run on true vs. estimated channels | Isolate the effect of estimation errors from the difference between two trainings | `scripts/make_same_model_csi_configs.py`, `scripts/same_model_csi_ratio.py` |
| Size generalization & depth ablation | MANET-GNN models of different depths / training sizes evaluated on the same test networks of one or more sizes (study configs list models × test sets) | Reproducible scripts for the size-generalization and depth-ablation figure | `scripts/SizeGeneralization.py`, `scripts/DepthAblation.py`, `utils/GNNStudyUtils.py`, `config/paper/Generalization/` |
| Convergecast (F4) models | GNN and FFN trained for convergecast (full and estimated CSI) | Dedicated F4 models and benchmark | `config/paper/Convergencecast/`, `scripts/fill_convergecast_paths.py` |

### Evaluation protocol

* **Test sets.** Fresh connected networks generated with QuaDRiGa (`5G-ALLSTAR_DenseUrban`, as the
  training data), from random-number streams disjoint from the training set: 500 networks for the main
  figures, plus the density, topology, and heterogeneous-radio sets above.
* **Estimated CSI.** Every method decides from the LMMSE estimate and is scored on the true channel, i.e.,
  the reported rate is the rate the allocation achieves. The SNR → noise mapping uses the true-channel
  power, so full- and estimated-CSI results share the noise level at each SNR.
* **Centralized optimizer (B1).** Started from the Greedy-Split, Widest-Path, and MANET-GNN allocations for
  every framework (AdamW, 1000 iterations, step size 0.1); the best iterate is kept, so it is never below
  any of its initializations. It can run on its own SNR grid (`optimizer snr`).
* **Centralized Widest-Path (B5).** Ties between routes with the same bottleneck gain are broken by the
  fewest hops (fewest edges for multicast), i.e., the shortest widest route, as in the decentralized
  Widest-Path (B6).
* **Results files** store per-network rates (`per_sample`), and the density results also store the ER
  edge probability of each network (`p_vec`).

### Model

* `CandidateRoutingHead.mask_dead_routes` (default on): candidate (band, route) options with a hop that
  does not support the band are excluded from the selection; if a band has no live route, the message is
  not routed over it, and its power moves to the other bands. Local, and an exact no-op on homogeneous
  networks.
* Heterogeneous-radio scoring: before scoring, each node drops power on unsupported channels and rescales
  the rest to the same total power (a no-op on homogeneous networks).

### Tooling

* `scripts/run_benchmarks_parallel.py`: runs all benchmark configs in parallel worker processes, balanced
  by estimated cost, skipping configs whose inputs are missing.
* `scripts/Optimizer_vs_GNN.py --configs ...`, and config paths on the command line for
  `scripts/Decentralized_MANET_Simulation.py` and `MANET_FFN/main.py`.
* `MANET_DATA_ROOT` environment variable for the data folder; `scripts/install_paper_configs.py` installs
  `config/paper/` into it.
* New benchmark config keys: `snr list`, `optimizer snr`, `optimizer lr`, `optimizer iterations`
  (see [docs/DESIGN.md §8](docs/DESIGN.md#8-config-key-reference)).
* `generate_graph_data` also loads `.pt` channel sets and uses the first `num samples` networks of a file.
* Benchmark figures plot each method at the SNRs it was evaluated at.
* `matlab/quadriga_demo.m` and `matlab/generate_quadriga_channels_with_scenario.m` updated to the
  versions that generated the training data (DenseUrban scenario; ER edge probability per sample).
