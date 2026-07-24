# MANET-GNN — System Model & Design Notes

This document expands the [README](../README.md) with the full system model, the MANET-GNN
architecture, the training objective, and the key algorithmic/engineering decisions behind the
implementation. It is the technical companion to the paper *MANET-GNN: Learned Decentralized
Optimization of Power Allocation in Multi-Channel MANETs* (under review, IEEE TCOM); equation
references below follow the revised manuscript.

> **Notation.** $\lVert\cdot\rVert_0$ is the $\ell_0$ pseudo-norm (number of non-zeros),
> $\lVert\cdot\rVert_F$ the Frobenius norm. $\mathcal V,\mathcal E$ are the node/edge sets of the
> MANET graph $\mathcal G=(\mathcal V,\mathcal E)$, $B$ the number of orthogonal channels, and $K$ the
> number of concurrent messages (commodities). The block index $t$ is dropped throughout, as
> allocation is solved per signal block.

---

## 1. System Model

### 1.1 Network and channels

A dynamic multi-hop multi-channel MANET is an **undirected, connected** graph
$\mathcal G=(\mathcal V,\mathcal E)$ with reciprocal links, **block-fading** (constant within a block,
independent across blocks). Each link carries up to $B$ orthogonal communication resources
(channels); the channel on link $(i,j)$ is a $B\times 1$ complex vector

```math
\mathbf h_{i\leftrightarrow j}=\big[h^{(1)}_{i\to j},\dots,h^{(B)}_{i\to j}\big]^\top\in\mathbb C^B,\qquad
h^{(b)}_{i\to j}=h^{(b)}_{j\to i}\ \text{(reciprocity)}.
```

Nodes may be **heterogeneous**: if node $i$ does not support resource $b$, then $h^{(b)}_{i\to j}=0$
for all $j$ and that link is removed on band $b$. Transmissions on different channels are orthogonal,
but **simultaneous transmissions on the same channel interfere**. With unit-variance symbol
$s^{(b)}_{i\to j}$ and allocated amplitude $p^{(b)}_{i\to j}$, node $j$ observes

```math
y^{(b)}_{i\to j}=h^{(b)}_{i\to j}\,p^{(b)}_{i\to j}\,s^{(b)}_{i\to j}+\underbrace{\textstyle\sum_{l\in\mathcal N_j\setminus\{i\}}h^{(b)}_{l\to j}\,p^{(b)}_{l\to j}\,s^{(b)}_{l\to j}}_{\text{co-channel interference }I^{(b)}_j}+\,w^{(b)}_j,\qquad w^{(b)}_j\sim\mathcal{CN}(0,\sigma_b^2).
```

### 1.2 Power tensor and feasible set

Power is stacked into a $B\times K\times|\mathcal V|\times|\mathcal V|$ tensor $\mathbf P$ with
$\sum_{k=1}^{K}[\mathbf P]_{b,k,i,j}=p^{(b)}_{i\to j}$. **Entries of $\mathbf P$ are transmission
amplitudes** — received signal power is proportional to $[\mathbf P]^2$. The feasible set enforces a
**per-node unit power budget** and **one message per channel** (orthogonality):

```math
\mathcal P=\Big\{\mathbf P\in[0,1]^{B\times K\times|\mathcal V|\times|\mathcal V|}\;:\;
\big\lVert[\mathbf P]_{:,:,i,:}\big\rVert_F\le 1\ \forall i,\;\;
\big\lVert[\mathbf P]_{b,:,i,j}\big\rVert_0\le 1\ \forall (i,j,b)\Big\}.
```

The Frobenius budget $\lVert[\mathbf P]_{:,:,i,:}\rVert_F=\sqrt{\sum_{b,k,j}P_{b,k,i,j}^2}\le 1$ is what
`utils/TensorUtils.py:normalize_power` projects onto (a **down-only** scaling — it never increases
power; see §5.4).

### 1.3 Rates (SINR) and the end-to-end objective

Under treating-interference-as-noise, the per-link achievable rate on channel $b$ for message $k$ is
an **SINR** expression — power enters squared (amplitude), and co-channel interference sits in the
denominator alongside noise:

```math
R^{(b)}_{i\to j,k}(\mathbf P)=\log_2\!\Big(1+\tfrac{|h^{(b)}_{i\to j}|^2\,[\mathbf P]^2_{b,k,i,j}}{\sigma_b^2+I^{(b)}_{i\to j}(\mathbf P)}\Big),\qquad
I^{(b)}_{i\to j}(\mathbf P)=\sum_{l\in\mathcal N_j}|h^{(b)}_{l\to j}|^2\sum_{k=1}^{K}\sum_{q\in\mathcal N_l}P^2_{b,k,l,q}.
```

A route is only as good as its weakest link, so over a connected subgraph $\psi\subseteq\mathcal E$ the
per-band rate is the **bottleneck**, and the bands are **summed** (each band picks its own best route):

```math
R^{(b)}_{\psi,k}(\mathbf P)=\min_{(i\to j)\in\psi}R^{(b)}_{i\to j,k}(\mathbf P),\qquad
R^{\text{E2E}}_k(\mathbf P;\Phi)=\sum_{b=1}^{B}\ \max_{\psi\in\Phi}\ R^{(b)}_{\psi,k}(\mathbf P).
```

The power allocation problem is the **max–min over messages** (fairness across commodities):

```math
\mathbf P^\star=\arg\max_{\mathbf P\in\mathcal P}\ \min_{k\in\{1,\dots,K\}}\ R^{\text{E2E}}_k(\mathbf P;\Phi_k).
```

$\Phi_k$ is built **offline** (simple source–destination paths via DFS for path-based frameworks; all
connected subgraphs spanning the source and receivers for multicast). It is used only for training-time
objective evaluation — **at inference MANET-GNN emits the allocation after a fixed number of rounds,
with no path/subgraph enumeration.**

> **Design consequence — band aggregation = SUM (§5.1).** The code reduces per-band rates with `sum`
> (not `mean`) in all objective modules, matching $R^{\text{E2E}}_k=\sum_b(\cdot)$.

---

## 2. Communication Frameworks

The same physical layer supports several traffic patterns, differing only in $K$ and $\{\Phi_k\}$. Each
maps to a `mode` string in the code.

| # | Framework | Code `mode` | $K$ | $\Phi_k$ |
|---|-----------|-------------|-----|----------|
| F1 | **Unicast** — one Tx → one Rx | `single` | 1 | all Tx→Rx paths |
| F2 | **Multicast** — one Tx → $Q$ Rx, common message | `multicast` | 1 | subgraphs spanning $\{u^{\mathrm{Rx}}_q\}$ |
| F3 | **Multicommodity** — one Tx → $K$ Rx, distinct messages | `multi` | $K$ | Tx→$u^{\mathrm{Rx}}_k$ paths |
| F4 | **Convergecast** — $K$ Tx → one Rx (inverse of F3) | `converge` | $K$ | $u^{\mathrm{Tx}}_k$→Rx paths |
| F5 | **Many-to-Many** — $K$ independent Tx/Rx pairs (most general) | `multiunicast` | $K$ | $u^{\mathrm{Tx}}_k$→$u^{\mathrm{Rx}}_k$ paths |

> **Naming note.** The paper's framework **F5 "Many-to-Many"** is the code's `mode = multiunicast`
> (and appears as `_multiunicast` in result filenames). They are the same framework: multiple
> source–destination pairs injecting independent traffic.

Single-message frameworks (F1, F2) use $K=1$. Multi-message frameworks (F3, F4, F5) are scored on the
**max–min (fair) reduction** — the objective that *forbids abandoning a commodity* — via
`reduce = fair` (§5.5).

---

## 3. Design Requirements

The learned optimizer must meet four requirements a centralized solver does not:

- **R1 — Decentralized:** node $i$ sets its own $\{p^{(b)}_{i\to j}\}$ from only its neighbors
  $\mathcal N_i$ and local (possibly noisy) CSI.
- **R2 — Bounded latency:** at most $L$ neighbor message exchanges.
- **R3 — Topology-generalizing:** one parameter set across graph sizes/topologies.
- **R4 — Robust to noisy CSI.**

---

## 4. MANET-GNN Architecture

A gated message-passing GNN whose depth *is* the optimizer's iteration count: each gated layer is one
round of decentralized refinement, and the number of rounds enforces the R2 latency budget.

### 4.1 Input encoding

- **Edge features** — real/imaginary parts of the $B$ channel gains, a $2B$-vector
  $\mathbf e^{(0)}_{i\to j}=[\,\mathrm{Re}\{h^{(1..B)}_{i\to j}\}\Vert\mathrm{Im}\{h^{(1..B)}_{i\to j}\}\,]^\top$.
- **Node features** — an equal-split power prior $p^{(b)}_{i\to j}\equiv 1/\sqrt{|\mathcal N_i|\,B}$
  stacked with a role vector $\mathbf r_i$ (Tx flag, Rx flag, relay, and $K$ commodity slots). Only the
  first $B$ entries are propagated as the initial node embedding; the rest are static side information.

### 4.2 Gated message-passing backbone

The backbone stacks `num_layers` gated GNN layers. **Each gated layer performs two neighbor exchanges**
(message pass + embedding broadcast), so the paper's number of communication rounds is
$L=2\cdot\texttt{num\_layers}$.

> ⚠️ **`L` convention.** The config key **`L` = `num_layers` = number of gated layers**
> (`scripts/*.py: num_layers=L`). The paper's $L$ is the number of *communication rounds*
> = $2\times$ the config `L`. The reported $|\mathcal V|=10$ experiments use **3 gated layers**
> (config `L=3`, the paper's $L=6$). The depth ablation sweeps the paper's $L\in\{2,4,6,10,14\}$.

Per layer $l$, each node updates edge and node embeddings locally (all hidden dims $=B$):

- **Edge update / gating** — an MLP on concatenated, layer-normalized endpoint and edge embeddings,
  with a sigmoid self-gate before the residual:

```math
\Delta\mathbf e^{(l)}_{i\to j}=\mathrm{MLP}^{(l)}_e\big(\bar{\mathbf e}^{(l-1)}_{i\to j}\Vert\bar{\mathbf x}^{(l-1)}_j\Vert\bar{\mathbf x}^{(l-1)}_i\big),\quad
\mathbf e^{(l)}_{i\to j}=\mathrm{LN}\big(\mathbf e^{(l-1)}_{i\to j}+\sigma(\Delta\mathbf e^{(l)}_{i\to j})\odot\Delta\mathbf e^{(l)}_{i\to j}\big).
```

- **FiLM-modulated message** — the edge embedding modulates the node-to-node message:

```math
\mathbf m^{(l)}_{i\to j}=\big(\mathbf 1+\boldsymbol\gamma^{(l)}_{i\to j}\big)\odot\mathbf W^{(l)}_{\mathrm{msg}}\bar{\mathbf x}^{(l-1)}_i+\boldsymbol\beta^{(l)}_{i\to j}\in\mathbb R^B,\quad
\boldsymbol\gamma=\mathbf W_\gamma\mathbf e^{(l)},\ \boldsymbol\beta=\mathbf W_\beta\mathbf e^{(l)}.
```

- **Aggregation** — mean of incoming messages through a node MLP, with residual + LayerNorm:

```math
\mathbf x^{(l)}_i=\mathrm{LN}\Big(\mathbf x^{(l-1)}_i+\mathrm{MLP}^{(l)}_a\big(\tfrac{1}{|\mathcal N_i|}\textstyle\sum_{j\in\mathcal N_i}\mathbf m^{(l)}_{j\to i}\big)\Big).
```

### 4.3 Decoder, routing head, and power projection

After the final layer, for each link $(i\to j)$ node $i$ decodes from
$\mathbf f_{i\to j}=[\,\mathbf e^{(L/2)}_{i\to j}\Vert\mathbf x^{(L/2)}_i\Vert\mathbf x^{(L/2)}_j\,]$:

1. **Power head** $\tilde{\mathbf p}_{i\to j}\in\mathbb R^B_+$ — an FC layer with **Softplus**.

2. **Candidate-based routing head** (multi-message F3–F5). Rather than independent per-edge masks,
   routing selects **complete routes** from a *local* candidate set of source→destination paths of
   length $\le L/2$ hops (the same set at train and inference — no train/test mismatch, no global route
   table). Each candidate $p_{k,m}$ gets a learned path embedding
   $\bar{\mathbf e}_{k,m}=\tfrac{1}{|p_{k,m}|}\sum_{(i,j)\in p_{k,m}}\mathbf e^{(l)}_{i\to j}$ concatenated
   with **band-dependent physical route features** $[\,g^{\min}_{b,k,m},\,g^{\mathrm{avg}}_{b,k,m},\,|p_{k,m}|,\,\eta_{b,k,m}\,]$
   (min / mean link gain, hop count, and receiver-side interference exposure
   $\eta_{b,k,m}=\sum_{(i,j)\in p_{k,m}}\sum_{q\in\mathcal N_j\setminus\{i\}}|h^{(b)}_{q\to j}|^2$). A shared
   MLP scores each candidate; a **per-band soft selection** (softmax at temperature $\tau_r$) is used in
   training and a **hard per-band argmax** at inference, projected onto edges:

```math
Z_{b,k,i,j}=\sum_{m}\pi_{b,k,m}\,\mathbf 1\{(i,j)\in p_{k,m}\},\qquad
\pi_{b,k,m}=\frac{e^{a_{b,k,m}/\tau_r}}{\sum_{m'}e^{a_{b,k,m'}/\tau_r}}.
```

   Since the argmax is per band, a commodity may use **different routes on different bands**
   (consistent with the band-sum in §1.3). For single-message frameworks the routing variable reduces
   to a per-edge message mask.

3. **Power projection** — mask the raw amplitude by the (square-root) routing variable and normalize to
   the per-node budget:

```math
[\mathbf P]_{:,k,i,j}=\frac{\sqrt{[\mathbf z_{i\to j,k}]}\odot\tilde{\mathbf p}_{i\to j,k}}{\sqrt{\sum_{k'=1}^{K}\sum_{j'\in\mathcal N_i}\big\lVert\sqrt{[\mathbf z_{i\to j',k'}]}\odot\tilde{\mathbf p}_{i\to j',k'}\big\rVert_2^2}}.
```

At inference each node runs only the forward pass over its $L/2$-hop neighborhood — the policy is fully
decentralized, and complexity is $\mathcal O(L|\mathcal E|BK)$ messages, **independent of the number of
feasible paths** (which the centralized baselines must enumerate).

---

## 5. Key Design Decisions

Stated as the *final* design; each is load-bearing.

### 5.1 Band aggregation = SUM over bands

All objective modules reduce per-band rates with `sum`, matching $R^{\text{E2E}}_k=\sum_b\max_\psi(\cdot)$.
`mean → sum` is a uniform $\times B$ rescale — **ordering-preserving and retrain-free**. It does not
"rescue" the widest-path baseline (which legitimately scores lower because it uses one band while greedy
uses all $B$); that ordering is correct, not a bug.

### 5.2 SNR → σ convention (train ≡ eval)

SNR is defined by the noise level, $\mathrm{SNR}^{(b)}=10\log_{10}(1/\sigma_b^2)$, i.e.
$\sigma_b=10^{-\mathrm{SNR}/20}$ for unit channel variance. **Training and benchmarking use the identical
mapping** (`scripts/Decentralized_MANET_Simulation.py`, `MANET_FFN/main.py`). Treating $\sigma$ as
$10^{-\mathrm{SNR}/10}$ (a std) squares the variance — doubling the effective dB, mis-placing every
rate-vs-SNR curve, and diverging to NaN at high SNR (gradients $\sim1/\sigma^2$).

### 5.3 Route concentration

The bottleneck objective admits a "transmit everywhere" local optimum that floods the network with
same-band self-interference. Concentration is handled two ways:

- **Single-message frameworks (F1–F2):** a **rate-scaled Hoyer sparsity penalty** on the per-edge
  amplitude $a_{i,j}=\sqrt{\sum_b P^2_{b,i,j}}$,
  $\mathcal L_{\text{sparse}}=\lambda_s\,|R|\,(\lVert\mathbf A\rVert_1/\lVert\mathbf A\rVert_2-1)$. The
  amplitude **sums over bands first**, so multi-band use of *one* route is not penalized — only spreading
  across *edges* is. The Hoyer term is $0$ for one active edge, $\sqrt E-1$ for $E$ equal edges, and is
  **scale-invariant** (no under-power bias); scaling by the detached rate keeps it a stable fraction of
  the objective across the 0–50 dB mix. Config key `sparsity` (default 0).
- **Multi-message frameworks (F3–F5):** concentration is **structural** — the candidate-based routing
  head (§4.3) restricts power to whole candidate routes, so a separate sparsity term is unnecessary.

### 5.4 Per-node budget projection is down-only

`normalize_power` scales each node's allocation to satisfy $\lVert[\mathbf P]_{:,:,i,:}\rVert_F\le 1$ but
**never increases** power. Training therefore learns to *saturate then back off* under interference
rather than push power up against a weak high-SNR gradient. (Historical note: the fix corrected a
transposed-adjacency mask that would have zeroed legal power on directed graphs; symmetric adjacencies
were unaffected.)

### 5.5 Fair (max–min) reduction for multi-message frameworks

F3/F4/F5 are scored on $\min_k R^{\text{E2E}}_k$, threaded end-to-end via `reduce = fair` through both
training and the centralized-optimizer wrapper. `mean`/`sum` reductions reward *abandoning* the hardest
commodity (serving one Rx, zeroing the rest); fair forbids this and is the reported objective. A separate
`val reduce` (default `fair`) lets model selection track the reported metric.

### 5.6 Fair scoring parity across all methods

Every method — learned and heuristic — is scored through the **same** interference-aware objective, with
`ignore_zero_edges = False` for all baselines (a sparse allocation is scored honestly — an unpowered edge
on a candidate route drives that route's `min` to 0 rather than being skipped), the **full candidate set**
(`find_all_paths` / `find_multicast_subgraphs`) for widest-path and greedy, and the centralized
widest-path powering the **entire** chosen path. This makes "greedy nearly matches MANET-GNN for unicast"
a statement about the *same* feasible set and scoring, not an evaluation artifact.

### 5.7 Centralized optimizer as a valid upper reference (B1)

The centralized AdamW optimizer for the objective uses **four initializations** — random, Greedy-Split,
Widest-Path, and MANET-GNN — runs **1000 iterations with a line search over five step sizes**, projects
onto the per-node constraints after each update, and **returns the best iterate by the $\tau=0$ (hard)
eval objective**. Multi-initialization guarantees the optimizer $\ge$ each of greedy / widest / GNN, so
it is a sound upper reference. Training-time soft-min/soft-max relaxations (temperatures
$\tau_{\min},\tau_{\max}$, annealed toward hard) provide gradients; eval is always the hard objective.

---

## 6. Training

MANET-GNN is trained **unsupervised** (Algorithm 2) — no ground-truth power labels. The rate is the
max–min objective on each layer's allocation, evaluated with the **SINR** rates of §1.3 (so training
accounts for both noise and co-channel interference):

```math
R^{(l)}_d(\theta)=\min_{k}R^{\text{E2E}}_k\big(\mathbf P^{(l)}(\mathcal G_d,\{h^{(b)}_{i\to j,d}\};\theta);\Phi_k\big).
```

The non-smooth `min` (bottleneck) and `max` (route selection) are replaced during training by
soft-min / soft-max relaxations at temperatures $\tau_{\min},\tau_{\max}$ (sharpened over training;
inference uses the hard operators). The total loss combines rate maximization, a **monotonicity**
regularizer (each optimizer "iteration"/layer should improve the rate by a margin $\delta$), and the
single-message **sparsity** term (§5.3):

```math
\mathcal L_D(\theta)=\underbrace{-\tfrac{1}{|D|}\textstyle\sum_d R^{(L/2)}_d}_{\text{rate}}
+\ \lambda_m\underbrace{\tfrac{1}{|D|L}\textstyle\sum_d\sum_{l=1}^{L/2-1}\max(\delta-\Delta R^{(l)}_d,0)}_{\text{monotonicity}}
+\ \lambda_s\,\mathcal L_{\text{sparse}}.
```

**Noisy-CSI-aware (adversarial) training (R4).** The forward pass sees an **LMMSE**-estimated channel
$\{\hat h^{(b)}_{i\to j}\}$ (AWGN observation model), but the loss is always evaluated on the *true* CSI —
so improvements are measured in real end-to-end rate under perturbed inputs. Toggled by `LMMSE estimation`.

**Setup (from the paper's experiments).** QuaDRiGa frequency-selective channels; random graphs with edge
probability $p\in\{0.1,\dots,0.5\}$; $|\mathcal V|=10$, $B=6$. 2000 topologies (1600 train / 400 val),
trained jointly over **SNR 0–50 dB in 5 dB steps**. **3 gated layers ($L=6$ rounds)**, 100 epochs, AdamW
($\text{lr}=5\times10^{-4}$, weight decay $3\times10^{-5}$), cosine schedule, dropout $0.2$.

---

## 7. Benchmarks & Confidence Intervals

`scripts/Optimizer_vs_GNN.py` sweeps 8 methods across SNR, all scored through the identical
interference-aware objective and feasible set (§5.6):

| Method | Description |
|--------|-------------|
| **MANET-GNN** | The proposed decentralized learned optimizer |
| **Centralized Optimizer** (B1) | AdamW, 4 inits (random / greedy / widest / GNN), 1000 iters, line-search — upper reference |
| **MANET-FFN** | Topology-blind MLP over the flattened channel tensor (no message passing) |
| **CWP / DWP** | Centralized / Decentralized **Widest-Path** ("Best Single Channel") |
| **CGS / DGR** | Centralized / Decentralized **Greedy-Split** (shortest route, equal power) |
| **Equal-Split** | Uniform power over feasible links and bands |

- **MANET-FFN** is a deliberately **topology-blind** baseline — no Tx/Rx input, so its train Tx/Rx **must**
  match the benchmark Tx/Rx (keep `FFN_single.ini` synced with the comparison config).

- **Confidence intervals.** All reported results are averaged over **500 independent test topologies**
  ($|\mathcal V|=10$, $B=6$; $K=4$ for multi-message frameworks). `evaluate_across_snr` records, per method
  per SNR, the mean, the standard error of the mean (`results["sem"]`,
  $\mathrm{SEM}=\mathrm{std}_{\text{ddof}=1}/\sqrt N$), and the test-set size (`results["n_test"]`).
  `visualization/GraphingAux.py:plot_mean_rate_vs_snr` renders **95% CIs** ($\pm1.96\cdot\mathrm{SEM}$,
  whiskers clipped on a log axis) for every method at every SNR and annotates $N$ on the title. Older
  pickles predating `sem` fall back to reconstructing it from stored per-sample diagnostics. This makes
  the stability of small inter-method gaps explicit.

**Results at a glance.** The centralized optimizer leads throughout. For **unicast (F1)**, Greedy-Split
slightly edges MANET-GNN (single-pair routing is near-trivial for greedy). For every **multi-message /
multi-destination** framework (**F2–F5**), MANET-GNN is the strongest method after the centralized
optimizer from moderate SNR onward — heuristics that pick short/high-bottleneck routes independently
cannot coordinate commodities competing for shared power and interference. Orderings are preserved under
estimated CSI.

---

## 8. Config Key Reference

Selected keys under `[Train Parameters]` (see the README for full examples):

| Key | Meaning |
|-----|---------|
| `mode` | Framework: `single` / `multicast` / `multi` / `converge` / `multiunicast` (= Many-to-Many) |
| `B`, `n` | Number of bands, number of nodes |
| `L` | Number of **gated layers** (= paper's $L/2$; total rounds $=2L$). Reported experiments: `L=3` |
| `tx`, `rx` | Transmitter / receiver node indices (comma-separated for multi) |
| `SNR` | SNR sweep (dB) — converted to $\sigma_b=10^{-\mathrm{SNR}/20}$ |
| `mono` | Monotonicity regularizer weight $\lambda_m$ |
| `sparsity` | Single-message edge-sparsity weight $\lambda_s$ (§5.3; default 0) |
| `reduce` / `val reduce` | Multi-message reduction: `fair` (max–min) vs `mean`/`sum` (§5.5) |
| `LMMSE estimation` | Noisy-CSI (LMMSE) training toggle (R4) |
| `dropout`, `lr`, `wd`, `epochs`, `num samples`, `grad batch` | Standard training hyperparameters |

---

*Cross-references:* implementation lives in `models/` (architecture + training loop),
`utils/` (objective, centralized baselines, tensor/power projection), `Multicast/` and
`Multicommodity/` (per-framework objectives), and `scripts/` (training + benchmarking).
