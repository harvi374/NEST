# Core Experiments (Elliptic Dataset)

This directory contains the primary experiments for the FeTA / EP-FedProto paper, run on the Elliptic Bitcoin dataset.

- `01_elliptic_data_models_baselines.ipynb` — Data preprocessing, baseline models (Centralized, Local-only, FedAvg, FedSage+, FedProto, FjORD).
- `02_elliptic_rank_aware_aggregation.ipynb` — Implementation and evaluation of the rank-aware dimension-wise prototype aggregator (EP-FedProto).
- `03_elliptic_ablations.ipynb` — Ablation studies (No-SSL, No-FedProx, Uniform-Agg, Tier Distributions).
- `04_elliptic_visualizations.ipynb` — Figure generation and final metrics reporting.
