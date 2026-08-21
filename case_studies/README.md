# Case Studies

This directory contains deep-dive structural case studies.

## Peeling Chains

An isolated structural study demonstrating how different federated aggregation strategies (FedAvg, FedProto, EP-FedProto) behave when specific structural money laundering topologies (like a "peeling chain") are entirely withheld from some clients.

- `01_setup_and_smoke.ipynb` — Deterministic data generation, graph partitioning, and smoke tests.
- `02_full_experiment.ipynb` — The main federated experiment run.
