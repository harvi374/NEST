# NEST — Nested Embeddings for Scalable Tiers in Privacy-Preserving AML

A federated graph learning framework that lets institutions with different hardware budgets (8/16/32/64-dim) train and share one aligned embedding space — via Matryoshka-style nested prototypes and Rank-Aware ContribAgg — instead of forcing a shared prototype dimensionality.

**Target venue:** CODS-COMAD '26 (Bengaluru, India)
**Dataset:** [Elliptic Bitcoin](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
**Requires:** Python 3.9+, PyTorch 2.0+, PyTorch Geometric 2.3+

---

## Table of Contents

- [What this repo is](#what-this-repo-is)
- [A note on naming](#a-note-on-naming)
- [Method summary](#method-summary)
- [Repository structure](#repository-structure)
- [Package API surface](#package-api-surface)
- [The `ExperimentConfig`](#the-experimentconfig)
- [Data pipeline](#data-pipeline)
- [Model architecture](#model-architecture)
- [Federated training loop](#federated-training-loop)
- [Aggregation strategies](#aggregation-strategies)
- [Losses](#losses)
- [Evaluation & explainability](#evaluation--explainability)
- [Notebooks](#notebooks)
- [Case studies](#case-studies)
- [External validation](#external-validation)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Reproducing the ablation table](#reproducing-the-ablation-table)
- [Results (paper)](#results-paper)
- [Limitations](#limitations)
- [Citation](#citation)

---

## What this repo is

This is the working codebase behind the NEST paper: a federated Anti-Money-Laundering (AML) detector for the Elliptic Bitcoin transaction graph that solves two problems simultaneously —

1. **Hardware heterogeneity** — clients (institutions) can only afford to share prototypes of dimension 8, 16, 32, or 64. NEST learns one 64-D embedding space where every smaller dimension is a *meaningful prefix* of the larger ones (Matryoshka Representation Learning), so no zero-padding or truncation loss is needed.
2. **Chronological non-IID drift** — clients see different time windows of transaction activity. FedProx-style proximal regularization on the shared encoder plus prototype re-alignment each round keeps clients coordinated.

It is organized as an installable Python package (`src/`) plus a set of driver notebooks — the notebooks are the actual entry points for running experiments; `src/` is the shared library they all import.

## A note on naming

The project has gone through several internal names as it evolved — you will see all of them in the code and docstrings, referring to the same lineage of work:

- **FeTA** — earliest framing (`src/__init__.py` docstring, package name `feta-aml` in `setup.py`)
- **PGFCL** — Prototype-Guided Federated Contrastive Learning (the main training loop is literally named `run_pgfcl` in `src/training/federated.py`)
- **EP-FedProto** — "Embedding-Prefix FedProto", used for the tiered/nested-budget variant (`supervised_round_client_ep`, the `trunc_dim` fixed-tier baseline in `FullGAT.forward`)
- **NEST** — the paper-facing name for the final nested-representation + Rank-Aware ContribAgg system described in the CODS-COMAD '26 submission

If you're cross-referencing the paper against the code: **the paper's NEST = this repo's PGFCL/EP-FedProto machinery**, specifically `rank_aware_contrib_agg()` + `multi_budget_proto_loss()` + `run_pgfcl()`.

---

## Method summary

```
Client tiers (d in {8, 16, 32, 64})
        |
        v
Shared encoder g_psi : SAGEConv -> SAGEConv -> GATv2Conv   ->  z in R^64
        |                                     (src/models/encoder.py)
        v
Per-class prototypes (degree-weighted mean embedding)   (compute_embedding_prototypes)
        |
        v
Nested prefixes phi_d(z) = z[:d], d in {8,16,32,64}         (multi_budget_proto_loss)
        |
        v
Local loss = focal/weighted-CE + lambda*L_proto + (mu/2)||psi-psi_global||^2   (supervised_round_client_ep)
        |
        v
Server: Rank-Aware ContribAgg -- aggregate each dimensional
segment [0,8) [8,16) [16,32) [32,64) using only clients whose
tier covers that segment, weighted by size x prototype cosine
similarity to the previous global prototype                       (rank_aware_contrib_agg)
        |
        v
FedAvg on encoder weights (contrib- or gradguard-weighted)         (fedavg_state_contrib)
        |
        v
FedPer: per-client head fine-tuning, encoder frozen, heads NOT averaged  (fedper_head_finetune)
```

A short SSL pre-warm phase (label-aware SupCon contrastive loss on the shared encoder, encoder-only FedAvg) runs *before* the main loop, never inside it.

---

## Repository structure

```
NEST-main/
├── README.md
├── requirements.txt
├── setup.py                          # package name: feta-aml (legacy, see naming note)
│
├── src/
│   ├── __init__.py
│   ├── config.py                     # ExperimentConfig -- single source of truth for hyperparameters
│   ├── utils.py                      # set_seed()
│   │
│   ├── models/
│   │   ├── encoder.py                # SAGEGATEncoder (SAGE x2 + GATv2Conv), ordered-dropout, attention export
│   │   ├── head.py                   # ClassHead -- 2-layer classification head
│   │   ├── full_gat.py               # FullGAT -- encoder + head, supports trunc_dim (tiered truncation)
│   │   └── sage_baseline.py          # SAGEModel -- pure GraphSAGE baseline
│   │
│   ├── aggregation/
│   │   ├── fedavg.py                 # fedavg_state_selective / _contrib / encoder_only
│   │   ├── contrib_agg.py            # ContribAgg + Rank-Aware ContribAgg + Uniform-Agg (ablation)
│   │   └── grad_guard.py             # GradGuard -- cosine-similarity anomaly filtering
│   │
│   ├── training/
│   │   ├── losses.py                 # weighted CE, focal, InfoNCE, SupCon, prototype & multi-budget losses
│   │   ├── ssl_pretrain.py           # contrastive SSL pre-warm phase
│   │   ├── supervised.py             # per-client local training, prototype inheritance, FedPer
│   │   └── federated.py              # run_pgfcl() -- the main 3-phase federated training loop
│   │
│   ├── data/
│   │   ├── elliptic.py               # load_elliptic() -- Elliptic loader + temporal-deviation features
│   │   ├── splits.py                 # temporal_federated_split() -- chronological client partitioning
│   │   ├── graph_utils.py            # local / inductive edge-index extraction
│   │   └── ieee_cis.py               # IEEE-CIS loader (stub -- not yet implemented)
│   │
│   └── evaluation/
│       ├── metrics.py                # evaluate_tuned() -- PR-tuned threshold, inductive eval, avg_metrics()
│       ├── calibration.py            # compute_ece() -- Expected Calibration Error
│       └── saliency.py               # extract_node_saliency() -- GAT attention-based node risk scoring
│
├── notebooks/                        # primary Elliptic experiments (see below)
├── case_studies/peeling_chain/       # cross-tier collaboration deep-dive
└── external_validation/              # IEEE-CIS generalization check
```

---

## Package API surface

Everything below is re-exported from `src/__init__.py` and its subpackage `__init__.py` files, so `from src import X` works directly for any of these:

| Category | Symbols |
|---|---|
| Config | `ExperimentConfig`, `config_hash` |
| Models | `SAGEGATEncoder` (alias `GATEncoder`), `ClassHead`, `FullGAT`, `SAGEModel` |
| Data | `load_elliptic`, `temporal_federated_split`, `make_mask`, `get_local_edge_index`, `get_inductive_edge_index` |
| Losses | `weighted_ce`, `focal_loss`, `supervised_loss`, `infonce_loss`, `label_aware_edge_supcon`, `prototype_supcon_loss`, `multi_budget_proto_loss` |
| Aggregation | `fedavg_state_selective`, `fedavg_state_contrib`, `fedavg_encoder_only`, `compute_client_contrib_weights`, `rank_aware_contrib_agg`, `uniform_dimwise_agg`, `compute_update_directions`, `grad_guard_weights` |
| Evaluation | `evaluate_tuned`, `avg_metrics`, `compute_ece`, `extract_node_saliency` |

---

## The `ExperimentConfig`

`src/config.py` defines a single dataclass that every notebook and script should import instead of redefining its own copy (an earlier failure mode the codebase explicitly fixes -- see the module docstring: *"Previously duplicated 6+ times across notebooks"*).

| Group | Key fields | Default |
|---|---|---|
| Architecture | `hidden`, `emb_dim`, `heads`, `dropout`, `head_dropout` | 128, 64, 4, 0.4, 0.3 |
| Federation | `n_clients`, `seeds`, `test_ratio` | 4, `(42,123,7,456,789)`, 0.3 |
| Features | `use_dev_features` | `True` (330-dim raw+deviation vs. 165-dim raw only) |
| Schedule | `global_rounds`, `head_finetune_rounds`, `ssl_pretrain_rounds`, `ssl_epochs`, `sup_epochs`, `lr`, `lr_min` | 100, 20, 6, 20, 25, 0.005, 0.0005 |
| FedProx | `mu_encoder`, `mu_head` | 0.01, 0.0 |
| SSL pre-warm only | `tau`, `ssl_edge_floor`, `feat_drop`, `edge_drop` | 0.7, 15, 0.3, 0.2 |
| Prototype aggregation | `lam_max`, `lam_warmup_rounds`, `ema_momentum`, `raw_blend`, `use_degree_weighting` | 0.6, 2, 0.85, 0.2, `True` |
| **Ablation switches** | `use_ssl`, `use_protos`, `use_fedprox` | `True, True, True` |
| **Contribution switches** | `use_contrib_agg`, `contrib_floor`, `use_grad_guard`, `anomaly_thresh`, `use_saliency`, `saliency_top_k`, `use_calibration`, `ece_bins` | `True, 0.1, True, -0.1, True, 20, True, 15` |
| Loss | `use_focal_loss`, `focal_gamma` | `True`, 1.0 |
| Label propagation | `use_label_prop`, `lp_alpha`, `lp_steps` | `False` (ablation showed it hurts), 0.9, 1 |
| Personalization | `use_fedper` | `True` (local-only head fine-tuning, heads never averaged) |
| Baselines | `baseline_rounds`, `baseline_local_epochs`, `centralized_epochs` | 50, 25, 150 |

`config_hash(cfg)` produces a short deterministic MD5 hash (excluding `seeds`) -- used to detect checkpoint key collisions between differently-configured runs.

---

## Data pipeline

**`load_elliptic()`** (`src/data/elliptic.py`)
- Loads `elliptic_txs_features.csv`, `elliptic_txs_edgelist.csv`, `elliptic_txs_classes.csv`.
- Auto-discovers the dataset path via `ELLIPTIC_DIR` env var, or by walking `/kaggle/input` (Kaggle-first design), falling back to `./data/elliptic` locally.
- Builds 165 raw features; if `use_dev_features=True`, appends 165 **per-timestep z-scored deviation features** (each feature normalized against the mean/std of *its own timestep*, so temporal drift is exposed as a signal rather than hidden) -> 330-dim total.
- Global `StandardScaler`, NaN/Inf sanitization, class remap (`'1'->1` illicit, `'2'->0` licit, `'unknown'->-1`).
- Returns a single PyG `Data` object with `x`, `edge_index`, `y`, and `timestep`.

**`temporal_federated_split()`** (`src/data/splits.py`)
- Sorts labeled nodes by timestep, splits into `n_clients` contiguous chronological chunks (`np.array_split`) -- this *is* the non-IID mechanism.
- Within each client, stratifies illicit/licit separately into train/test by `test_ratio`, so class balance per split is preserved even though absolute counts differ wildly across clients.

**`graph_utils.py`**
- `get_local_edge_index` -- transductive: both endpoints inside the mask (used for local client training).
- `get_inductive_edge_index` -- at least one endpoint in the test set (used for evaluation, so test nodes never contaminate training).

---

## Model architecture

**`SAGEGATEncoder`** (`src/models/encoder.py`)
```
x -> SAGEConv -> +skip -> BN -> ReLU -> dropout   (h1)
  -> SAGEConv -> +skip -> BN -> ReLU -> dropout   (h2)
  -> GATv2Conv(heads=1, concat=False)              -> z in R^emb_dim (default 64)
```
- `forward_with_attention()` -- returns GATv2 attention weights at zero extra compute cost, feeding directly into `extract_node_saliency()`.
- `forward_od(x, edge_index, width_ratio)` -- **Ordered Dropout** (FjORD-style): zeroes the tail channels of every hidden/output layer beyond `width_ratio x layer_width`, in fixed channel order, so a low-budget client's active sub-network is always a strict nested prefix of every larger client's -- this is the encoder-side counterpart to the paper's prototype-side nesting.
- `encode_with_proj()` -- adds a small MLP projection head, used only during SSL pre-training (InfoNCE/SupCon).

**`ClassHead`** -- 2-layer MLP (`in_dim -> max(128, 2*in_dim) -> half -> n_classes`), BatchNorm + dropout.

**`FullGAT`** (`src/models/full_gat.py`) -- composes encoder + head. Its `forward(..., trunc_dim=None)` argument is the fixed-tier baseline mechanism: when set, it zero-masks embedding dimensions `>= trunc_dim` **before** the classification head -- i.e. the literal implementation of the prefix operator `phi_d(z) = z[:d]` from the paper, used by the EP-FedProto v3 fixed-tier comparison.

**`SAGEModel`** -- pure GraphSAGE baseline at matched dimensions, for fair comparison against the GATv2-based encoder.

---

## Federated training loop

**`run_pgfcl()`** (`src/training/federated.py`) is the main orchestrator, run once per seed:

1. **SSL pre-warm** (`ssl_pretrain_phase`, gated by `cfg.use_ssl`) -- `ssl_pretrain_rounds` rounds of label-aware SupCon contrastive training per client, encoder-only FedAvg between rounds. Never re-entered once the main loop starts.
2. **Main loop** (`cfg.global_rounds`, default 100):
   - Encode all clients' data with the current global encoder (no grad) -> per-client, per-class **degree-weighted prototypes** (`compute_embedding_prototypes`).
   - **Chain prototype inheritance** (`chain_prototype_inheritance`) -- each client's prototype is EMA-blended with the *previous* client's blended prototype (`ema_momentum=0.85` for the illicit class, fixed 0.3 for licit), forming a running consensus rather than an independent per-round mean.
   - Prototype drift (`||delta_p||_2`) is logged per round for both classes.
   - Each client trains locally (`supervised_round_client`) for `sup_epochs` with: focal/weighted-CE loss + FedProx proximal term on the **encoder only** (head is exempt) + prototype-alignment loss weighted by a linearly-warmed-up `lam` (0 -> `lam_max` over `lam_warmup_rounds`).
   - Server aggregation: **GradGuard** takes priority if enabled, else **ContribAgg**, else plain size-weighted FedAvg (see Aggregation strategies below).
   - Global model evaluated every round (`evaluate_tuned`); best-F1 checkpoint kept.
   - Every 10 rounds (if `use_saliency`): attention-based node saliency extracted and logged.
3. **FedPer head fine-tuning** (`fedper_head_finetune`, `head_finetune_rounds=20`) -- encoder frozen, each client fine-tunes its **own** head locally; heads are never averaged back. Final metric = mean +/- std across clients' personalized heads.
4. **Calibration** (`compute_ece`, if `use_calibration`) computed once on the final model.

Every run's result dict includes `config_hash`, per-round `f1_curve`, prototype-generation timing, and (if FedPer ran) per-client fine-tuned results.

---

## Aggregation strategies

`src/aggregation/`

| Function | File | What it does |
|---|---|---|
| `fedavg_state_selective` | `fedavg.py` | Plain size-weighted FedAvg over the full state dict |
| `fedavg_state_contrib` | `fedavg.py` | FedAvg using externally supplied weights (from ContribAgg or GradGuard) -- this is what the main loop actually calls |
| `fedavg_encoder_only` | `fedavg.py` | Averages only `encoder.*` params -- used during SSL pre-warm |
| `compute_client_contrib_weights` | `contrib_agg.py` | **ContribAgg** -- weight = `sqrt(client size) x prototype_quality`, where quality is a floor-clamped rescaling of cosine similarity between the client's illicit prototype and the previous global illicit prototype |
| `rank_aware_contrib_agg` | `contrib_agg.py` | **Rank-Aware ContribAgg** (the paper's core aggregator) -- aggregates each dimensional segment `[0,8) [8,16) [16,32) [32,64)` independently; only clients whose tier dimension covers a segment are eligible for it; within a segment, weighting is the same size x quality scheme as ContribAgg but computed **only on that segment's slice** of the prototype |
| `uniform_dimwise_agg` | `contrib_agg.py` | Ablation arm ("Uniform-Agg", used in `03_elliptic_ablations.ipynb`) -- same segment-wise eligibility, but pure size-proportional weighting, no cosine-similarity quality term |
| `grad_guard_weights` / `compute_update_directions` | `grad_guard.py` | **GradGuard** -- flattens each client's full parameter delta vs. global, computes cosine similarity to the size-weighted average update direction, and **excludes entirely** (weight -> 0, not just down-weighted) any client below `anomaly_thresh` (default `-0.1`) |

Note the priority order in `run_pgfcl`: **GradGuard overrides ContribAgg** when both are enabled (`if cfg.use_grad_guard: ... elif cfg.use_contrib_agg: ...`) -- they are not currently combined additively.

---

## Losses

`src/training/losses.py`

- **`weighted_ce`** -- inverse-class-frequency-weighted cross-entropy, weights clamped to `[0.1, 10.0]`.
- **`focal_loss`** -- Lin et al. 2017 focal loss with the same inverse-frequency `alpha`, `gamma=cfg.focal_gamma` (default 1.0). `supervised_loss()` dispatches between the two based on `cfg.use_focal_loss`.
- **`infonce_loss`** -- standard InfoNCE with feature dropout + edge dropout augmentation; negatives sampled from local (client) nodes excluding the current batch's anchors.
- **`label_aware_edge_supcon`** -- the loss actually used during SSL pre-warm: illicit nodes are anchors, other illicit nodes are positives, licit nodes are negatives (a supervised-contrastive variant that pre-shapes the embedding space around the minority class before any classifier is trained).
- **`prototype_supcon_loss`** -- cross-entropy between a node's normalized embedding and the two (licit/illicit) global prototypes treated as a 2-way softmax -- this *is* NEST's core nested-prototype-alignment objective at a single fixed dimension.
- **`multi_budget_proto_loss`** -- wraps the above across every budget in `(8, 16, 32, 64)` (or a client's `own_budgets` subset in the EP-variant), equally weighted by default -- the direct implementation of the paper's Eq. 13 multi-budget objective L_proto.

---

## Evaluation & explainability

- **`evaluate_tuned`** (`evaluation/metrics.py`) -- tunes the decision threshold on the **train** set's precision-recall curve (no test leakage), then evaluates **inductively** on test edges (`get_inductive_edge_index`). Returns accuracy, F1, AUC, precision, recall, confusion matrix, and the tuned threshold.
- **`avg_metrics`** -- mean +/- std across a list of per-client metric dicts (used for FedPer's personalized-head aggregate).
- **`compute_ece`** (`evaluation/calibration.py`) -- standard binned Expected Calibration Error (`cfg.ece_bins=15`), returns both the scalar ECE and per-bin `(confidence, accuracy, count)` triples for reliability-diagram plotting.
- **`extract_node_saliency`** (`evaluation/saliency.py`) -- pulls GATv2 attention weights via `forward_with_attention`, averages incoming attention per node, computes `risk_score = predicted_prob x mean_attention`, and returns the top-k highest-risk test nodes with their true labels -- used to distinguish structurally *central* vs. *peripheral* flagged transactions.

---

## Notebooks

`notebooks/` -- the primary Elliptic experiment suite, run in order:

1. **`01_elliptic_data_models_baselines.ipynb`** -- data preprocessing; baseline models (Centralized, Local-only, FedAvg, FedSage+, FedProto, FjORD).
2. **`02_elliptic_rank_aware_aggregation.ipynb`** -- implementation and evaluation of the rank-aware, dimension-wise prototype aggregator (EP-FedProto / NEST proper).
3. **`03_elliptic_ablations.ipynb`** -- ablation studies: No-SSL, No-FedProx, Uniform-Agg, tier-distribution sweeps.
4. **`04_elliptic_visualizations.ipynb`** -- figure generation and final metrics reporting (this is where the paper's Figures 3-9 come from).

## Case studies

`case_studies/peeling_chain/` -- a deep-dive structural study, **not** a claim about ground-truth laundering topology (the notebook itself flags this: *"a peeling-chain-like structural illustration, not a claim that the Elliptic labels establish..."*).

- **`01_setup_and_smoke.ipynb`** -- builds a deterministic, fixed-seed (42) case-study subgraph, validates the 64<->8 nested-prototype mechanics, runs a short NEST smoke test. Read-only w.r.t. the Kaggle input; writes a frozen graph + manifest for NB2.
- **`02_full_experiment.ipynb`** -- runs LocalOnly, FedProto-64, FedProto-8, and NEST(64<->8) on the frozen graph from NB1, and verifies embedding alignment via cosine similarity. This is the source of the paper's Section 4.6 Cross-Tier Collaboration Case Study and Figure 7 UMAP visualization.

## External validation

`external_validation/` -- generalization check beyond the primary Elliptic graph, using the **IEEE-CIS Fraud Detection** dataset:

- **`ieee_cis_centralized.ipynb`** -- centralized baseline.
- **`ieee_cis_federated.ipynb`** -- federated (NEST) experiments.

Note: `src/data/ieee_cis.py`'s `load_and_build_graph()` is currently a **stub** (`# TODO: Migrate IEEE-CIS specific graph loading logic here`) -- the graph-construction logic for this dataset currently lives inline in the two notebooks above rather than in the shared library.

---

## Installation

```bash
git clone <this-repo>
cd NEST-main
pip install -r requirements.txt
pip install -e .          # installs the package as "feta-aml" (see naming note)
```

`requirements.txt`:
```
torch>=2.0.0
torch-geometric>=2.3.0
torch-scatter
torch-sparse
torch-cluster
torch-spline-conv
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
scipy>=1.10
matplotlib>=3.7
seaborn>=0.12
shap>=0.42
networkx>=3.1
tqdm>=4.65
pyyaml>=6.0
nbformat>=5.0
psutil>=5.8.0
```

**Dataset**: download the [Elliptic Bitcoin dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) and either:
- run on Kaggle directly (the loader auto-discovers `/kaggle/input/**/elliptic_txs_features.csv`), or
- set `ELLIPTIC_DIR=/path/to/elliptic_bitcoin_dataset` locally, containing `elliptic_txs_features.csv`, `elliptic_txs_edgelist.csv`, `elliptic_txs_classes.csv`.

---

## Quickstart

```python
import torch
from src import (
    ExperimentConfig, load_elliptic, temporal_federated_split,
)
from src.training.federated import run_pgfcl

cfg = ExperimentConfig()                 # defaults: 4 clients, 100 rounds, tiers via trunc_dim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = load_elliptic(use_dev_features=cfg.use_dev_features)   # 330-dim (raw + temporal-deviation)
clients = temporal_federated_split(data, cfg)                  # 4 chronological, stratified splits

best_metrics, drift_curve, f1_curve, saliency_history = run_pgfcl(
    data, clients, device, cfg, seed=42, label='NEST'
)

print(f"F1={best_metrics['f1']:.4f}  AUC={best_metrics['auc']:.4f}  "
      f"Recall={best_metrics['rec']:.4f}  ECE={best_metrics.get('ece', float('nan')):.4f}")
```

To run a fixed-tier (non-nested) baseline for comparison, pass `proto_dim` (e.g. `proto_dim=8`) to `run_pgfcl` -- this routes through `FullGAT`'s `trunc_dim` masking so the model only ever sees an 8-dimensional embedding, end to end.

---

## Reproducing the ablation table

Each row of the paper's ablation table (Table 3) corresponds to flipping one `ExperimentConfig` flag before calling `run_pgfcl`:

| Paper row | Config change |
|---|---|
| No-FedProx | `cfg.use_fedprox = False` |
| No-EMA Aggregation | reduce/zero `cfg.ema_momentum` (affects `chain_prototype_inheritance`'s blending) |
| No-ContribAgg (Uniform) | swap `rank_aware_contrib_agg` for `uniform_dimwise_agg` in the aggregation call, or set `cfg.use_contrib_agg = False` for plain size-weighted FedAvg |
| No-FocalLoss (CE Only) | `cfg.use_focal_loss = False` (falls back to `weighted_ce`) |

`03_elliptic_ablations.ipynb` runs all arms with the same seeds (`cfg.seeds`) for direct comparison.

---

## Results (paper)

Elliptic Bitcoin, 5 seeds, 4 chronological clients:

| Method | F1 (higher better) | AUC (higher better) | Precision | Recall (higher better) |
|---|---|---|---|---|
| Local Only | 0.4617 | 0.9276 | 0.8131 | 0.4565 |
| FedAvg | 0.7475 | 0.9394 | 0.8871 | 0.6463 |
| FedProto | 0.7594 | 0.9440 | 0.9162 | 0.6488 |
| MOON | 0.7565 | 0.9396 | 0.8602 | 0.6760 |
| SCAFFOLD | 0.7023 | 0.9161 | 0.8445 | 0.6021 |
| **NEST (Ours)** | **0.7872** | **0.9550** | 0.8894 | **0.7065** |
| Central (upper bound) | 0.8258 | 0.9638 | 0.9016 | 0.7619 |

- Statistical significance vs. FedProto: Cohen's d = 1.77, p < 0.05 (paired, 5 seeds).
- Communication reduction at d=8 vs. full d=64: **87.5%** (64 B vs. 512 B per class-pair prototype), with near-parity F1/AUC/Precision/Recall retained (edge-constrained variant).
- Cross-tier case study (peeling-chain, Section 4.6): NEST 64->8 Client-B F1 of 0.5288/0.6958/0.6967 across 3 seeds vs. FedProto-8's 0.4919/0.5848/**0.2847** -- FedProto-8 collapses on one seed; NEST doesn't.
- Ablation deltas: removing FedProx costs the most (delta-F1 = -0.040); EMA, ContribAgg, and focal loss each cost roughly -0.003 to -0.009 individually.

Full per-seed distributions, confusion matrices, drift/convergence curves, loss-landscape visualizations, and SHAP/GradCAM attribution are generated in `notebooks/04_elliptic_visualizations.ipynb`.

---

## Limitations

- Rank-Aware ContribAgg needs enough participating clients to populate high-dimensional segments -- thin coverage at d=32/64 weakens those segments.
- Gains are largest on **chronologically structured** graphs (Elliptic); on the larger, less temporally structured IEEE-CIS graph, FedProto-8 remains competitive on F1/AUC -- the tiered design is most suitable where federated data has a genuine temporal component.
- `src/data/ieee_cis.py` is a stub; IEEE-CIS graph construction currently lives only in the external-validation notebooks.
- Representation budgets are fixed at `{8, 16, 32, 64}` rather than dynamically selected per round.
- GradGuard and ContribAgg are mutually exclusive in the current loop (GradGuard takes priority), not combined.

---

## Citation

```bibtex
@inproceedings{nest2026,
  title     = {NEST: Nested Embeddings for Scalable Tiers in Privacy-Preserving AML},
  booktitle = {Proceedings of the 9th Joint International Conference on Data Science and Management of Data (CODS-COMAD '26)},
  year      = {2026},
  address   = {Bengaluru, India},
  publisher = {ACM}
}
```

**License:** not yet specified in this repository -- add one before publishing the code publicly.
