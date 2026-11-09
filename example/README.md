# CycleAlign example

This example reproduces the CycleAlign result on the **Kowalczyk 2015** dataset
(mouse haematopoietic stem cells, `1,434` cells × `94` genes, physically sorted
into G1/S/G2M fractions; GEO `GSE65525`).

## Data

The `data/` directory contains:

- `mesc-tpm.txt` — TPM expression matrix (genes × cells, tab-separated).
- `labels.txt` — one reference stage label per cell (`1`=G1, `2`=S, `3`=G2/M).
- `panel.txt` — the 94-gene panel used for this dataset.

## Run

From the repository root:

```bash
# full configuration (30 seeds, Conv encoder)
bash example/run_example.sh

# quick smoke test (5 seeds, 200 epochs) — a few minutes on CPU
RUNS=5 EPOCHS=200 bash example/run_example.sh

# MLP encoder instead of Conv
ENCODER=mlp bash example/run_example.sh
```

The script writes results to `example/output_<encoder>_r<runs>/`, containing
`consensus_phase.csv`, `metrics.json`, and `theta_runs.npy` (see the top-level
[README](../README.md#output-files) for details).

## Expected result

With the default configuration (Conv, 30 seeds, `k=8` smoothing), the ensemble
consensus phase reaches a circular adjusted Rand index of approximately
**`ARI_circle ≈ 0.85`** against the sorted G1/S/G2M labels. The MLP encoder
reaches a comparable value (≈ 0.84). Exact values vary slightly with hardware
(see the "numerical reproducibility" note in the top-level README).
