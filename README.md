# CycleAlign

**A cycle-phase-aligned neural ensemble for robust unsupervised inference of cell-cycle pseudotime from single-cell transcriptomes**

CycleAlign infers a continuous, circular cell-cycle pseudotime for every cell from single-cell RNA-sequencing (scRNA-seq) data, without any labels and without a fixed reference gene set. It trains a compact phase-encoding neural network — either a one-dimensional convolutional (Conv) encoder or a fully connected (MLP) encoder — to map a panel of cycle genes to a scalar phase angle through a circular reconstruction objective, and then aligns a multi-seed ensemble of independently trained models by jointly estimating an optimal rotation and reflection for each model. The result is a reproducible consensus phase that is invariant to the arbitrary orientation of individual runs and robust to the "collapse" of individual models.

> **Note on naming.** The method was developed and is referred to in earlier internal materials as *SPAE-Ens* (multi-seed phase-alignment ensemble). The manuscript name is **CycleAlign**. The two names refer to the same method; the command-line tool in this repository retains the original file name for continuity.

---

## Table of contents

1. [Method overview](#method-overview)
2. [Repository layout](#repository-layout)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Quick start](#quick-start)
6. [Input data format](#input-data-format)
7. [Reproducing the paper's results](#reproducing-the-papers-results)
8. [Command-line reference](#command-line-reference)
9. [Output files](#output-files)
10. [Citation](#citation)
11. [License](#license)

---

## Method overview

CycleAlign consists of four steps (S1–S4):

1. **S1 — Preprocessing.** Read the TPM expression matrix, apply `log2(1 + x)`, select a gene panel, and z-score each gene across cells.
2. **S2 — Multi-seed training.** Train `R` independent autoencoders, each with a circular (sin/cos) output unit in the bottleneck that encodes the phase angle, from different random seeds while keeping the architecture and all hyper-parameters fixed.
3. **S3 — Circular alignment (core).** A phase is only meaningful up to rotation and reflection, so it cannot be naively averaged. Within the closed transformation set `{ θ → s·θ + φ | s ∈ {+1, −1}, φ ∈ [0, 2π) }`, find for each run the transformation that maximizes the mean resultant length `R̄` of the aligned phases, and iteratively re-align against the circular consensus of all runs until convergence.
4. **S4 — Consensus aggregation.** For each cell, compute the complex resultant vector `z_j = (1/R) Σ_r exp(i·θ_j^(r))` and take its argument `arg(z_j)` as the consensus phase (pseudotime).

Two optional, documented extensions are built into the tool:

- **Conv encoder** (`--encoder conv`, default): replaces the MLP with one-dimensional convolutions along a gene axis ordered by the optimal leaf order of a hierarchical clustering of gene-expression profiles (`Conv1D(8, k=7, tanh) → Dropout → Conv1D(2, k=5) → Flatten → Dense(h, tanh) → Dense(1)`). This is fully unsupervised (the leaf order uses no labels).
- **kNN phase smoothing** (`--smooth k`, default 8): an optional post-processing step that shrinks each cell's phase toward its `k` nearest neighbours in expression space. Set `--smooth 0` to disable.

---

## Repository layout

```
.
├── Multi-seed_phase_alignment_conv_reg.py   # main method (self-contained)
├── example/
│   ├── run_example.sh                       # one-command example runner
│   └── README.md                            # example data description
├── requirements.txt                         # Python dependencies
├── LICENSE
└── README.md
```

The main script is self-contained: it performs preprocessing, training, alignment, consensus aggregation, and evaluation, and writes all results to disk.

---

## Requirements

- Python 3.8–3.10
- TensorFlow 2.13 (CPU is sufficient; no GPU required)
- NumPy, pandas, scikit-learn, SciPy

The exact versions tested are listed in [`requirements.txt`](requirements.txt).

---

## Installation

We recommend an isolated environment via Anaconda/Miniconda:

```bash
# 1. create and activate a clean environment
conda create -n cyclealign python=3.9 -y
conda activate cyclealign

# 2. install dependencies
pip install -r requirements.txt
```

If you prefer `pip` only:

```bash
python -m venv cyclealign && source cyclealign/bin/activate
pip install -r requirements.txt
```

---

## Quick start

Run CycleAlign on a dataset directory that contains a TPM expression matrix and reference stage labels:

```bash
python Multi-seed_phase_alignment_conv_reg.py \
    --data-dir /path/to/your/dataset \
    --runs 30 \
    --encoder conv \
    --gene-order leaf \
    --outdir ./my_output
```

A minimal end-to-end example is provided in [`example/`](example/README.md):

```bash
bash example/run_example.sh
```

---

## Input data format

The script expects a dataset directory (passed via `--data-dir`) containing:

| File            | Format                                                                                     |
|-----------------|--------------------------------------------------------------------------------------------|
| `mesc-tpm.txt`  | Tab-separated TPM matrix, **genes × cells**. First row = cell names; first column = gene symbols. |
| `labels.txt`    | One integer per line (one per cell, same order as the matrix columns): `1`=G1, `2`=S, `3`=G2/M. |

The gene panel defaults to a built-in 306-gene panel (identical to the panel shipped with the SPAE reference implementation). To use your own panel, pass `--genes panel.txt`, where `panel.txt` lists one gene symbol per line.

Example matrix header:

```
gene	young_LT_HSC_2	young_LT_HSC_6	...
Chaf1b	1.170	263.270	...
```

---

## Reproducing the paper's results

The paper evaluates CycleAlign on seven public datasets. All are publicly available:

| Dataset | Accession | Cells | Type |
|---------|-----------|-------|------|
| E-MTAB-2805 (mouse mESC) | ArrayExpress | 288 | sorted G1/S/G2M |
| Leng 2015 (human hESC) | GEO GSE64016 | 247 | sorted G1/S/G2M |
| Kowalczyk 2015 (mouse HSC) | GEO GSE65525 | 1,434 | sorted G1/S/G2M |
| GSE121265 (human iPSC-FUCCI) | GEO | 888 | continuous FUCCI |
| GSE146773 (human U2OS-FUCCI) | GEO | 1,067 | continuous FUCCI |
| GSE125416 | GEO | — | — |
| Riba 2019 (mouse mESC, 10x) | Zenodo 10.5281/zenodo.10467234 | — | continuous |

For each dataset, prepare a directory with `mesc-tpm.txt` and `labels.txt` (for sorted datasets) or the equivalent continuous-phase reference (for FUCCI datasets), then run:

```bash
# Conv encoder (used for the main results)
python Multi-seed_phase_alignment_conv_reg.py \
    --data-dir <dataset_dir> --runs 30 --encoder conv --gene-order leaf \
    --smooth 8 --outdir <dataset>_conv30

# MLP encoder
python Multi-seed_phase_alignment_conv_reg.py \
    --data-dir <dataset_dir> --runs 30 --encoder mlp --smooth 8 \
    --outdir <dataset>_mlp30
```

The evaluation metrics reported in the paper — the circular adjusted Rand index (`ARI_circle`) against discrete G1/S/G2M labels, and the Jammalamadaka–Sarma circular–circular correlation (`ρ`) against continuous phase — are computed and printed automatically, and stored in `metrics.json` in the output directory.

### Note on numerical reproducibility

Identical seeds, environment, and TensorFlow CPU instruction set reproduce results bit-for-bit. Because TensorFlow's CPU float reduction order can vary with thread scheduling across machines, results may differ by a small amount on different hardware. The method forces single-threaded execution, disables oneDNN, and uses explicit per-operation seeds for all stochastic layers to maximize determinism (see the header comment of the main script for details). The default seed set (`--seed0 100`, seeds 100–129) is the fixed implementation whose output is reported in the paper.

---

## Command-line reference

Run `python Multi-seed_phase_alignment_conv_reg.py --help` for the full list. Key options:

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` | — | Dataset directory containing `mesc-tpm.txt` and `labels.txt` |
| `--expr` / `--labels` | — | Override individual file paths |
| `--genes` | built-in 306 | Path to a custom gene-panel file (one symbol per line) |
| `--runs` | 30 | Number of independent seeds `R` |
| `--seed0` | 100 | Starting random seed |
| `--epochs` | 1000 | Training epochs per seed |
| `--lr` | 2e-4 | Adam learning rate |
| `--width` | 30,20 | Hidden-layer widths (comma-separated) |
| `--encoder` | conv | `conv` (1-D conv) or `mlp` (fully connected) |
| `--gene-order` | leaf | Gene-axis order for the conv encoder: `leaf` (hierarchical-clustering optimal leaf order) or `orig` |
| `--circ` | 2 | Number of circular output units |
| `--bypass` | 3 | Bypass units (0 disables) |
| `--dropout` | 0.1 | Dropout rate |
| `--batch` | 0 (auto) | Batch size; 0 auto-selects a divisor of the cell count |
| `--iters` | 10 | Max re-alignment rounds |
| `--smooth` | 8 | kNN smoothing neighbourhood (`0` disables) |
| `--outdir` | ./spae_conv_output | Output directory |

---

## Output files

For each run, the output directory contains:

- `consensus_phase.csv` — per-cell consensus phase (radians, cycle fraction, reference stage), plus a smoothed phase column when `--smooth > 0`.
- `metrics.json` — per-run and ensemble metrics (`ARI_circle`, mean resultant length `R`, angular span, `ρ`, per-stage mean angles, half-split reproducibility).
- `theta_runs.npy` — the `R × N` matrix of candidate phases (for re-analysis).

The terminal also prints a per-run summary, the single-run distribution, the ensemble result, and a split-half reproducibility check.

---

## Citation

If you use CycleAlign in your research, please cite:

> Zhou, S., Liu, C., Wang, Q., Zang, M., & Liu, T. CycleAlign: a cycle-phase-aligned neural ensemble enables robust unsupervised inference of cell-cycle pseudotime from single-cell transcriptomes. *Nature Communications* (in submission).

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

Shusen Zhou — zhoushusen@ldu.edu.cn  
Ludong University, Yantai, China
