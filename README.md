# Maritime GSA Reliability Analysis

This repository contains the reproducibility code for a RESS study of reliable
multimodal federated learning in maritime safety systems. The layout follows
the supplied GitHub example: a concise repository root, one input directory,
dependency declaration, and an explicit experiment workflow.

Preliminary pilots, parameter-search programs, intermediate merge utilities,
superseded implementations, and manuscript-generation helpers are excluded.

## Environment

- Python 3.10 or newer (Python 3.11 recommended)
- Windows, Linux, or macOS; CUDA is optional

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

For HRRR GRIB2 data, installing `cfgrib` and `eccodes` with Conda is usually
more reliable than using pip alone.

## Repository structure

```text
maritime-gsa-reliability-analysis/
├── input/                          # original AIS, HRRR and EU MRV files
├── experiments/
│   ├── gsa/                        # primary experiments
│   └── supplementary/              # supplementary analyses
├── align_shipping_data.py          # raw-data alignment
├── requirements.txt
└── README.md
```

## Reproduction workflow

Run every command from this repository root.

### 1. Align the four data modalities

```bash
python align_shipping_data.py --data-dir input --max-ships 0
python -m experiments.gsa.prepare_dataset
```

This creates `aligned_samples.csv`, its quality report, and
`multimodal_dataset.csv`. Data preprocessing is fitted on training data
only, and vessel identities are kept disjoint across train/test partitions.

### 2. Main study

```bash
python -m experiments.gsa.run_ablation
```

The experiment includes the paper's FedAvg/FedBN/GSA ablations and uses the
same split within each random seed for paired comparison.

### 3. Reliability, privacy and downstream analyses

```bash
python -m experiments.gsa.robustness
python -m experiments.gsa.privacy_mia_disjoint
python -m experiments.gsa.gradient_inversion
python -m experiments.gsa.privacy_risk_analysis
python -m experiments.gsa.gradient_space_analysis
python -m experiments.gsa.communication_efficiency
python -m experiments.gsa.classification_and_link_attack
python -m experiments.gsa.downstream_emissions
```

### 4. Trajectory task

```bash
python -m experiments.gsa.build_trajectory_sequences
python -m experiments.gsa.trajectory_prediction_v2
```

`trajectory_prediction.py` provides shared model and data utilities imported by
`trajectory_prediction_v2.py`; it is not a separate experiment entry point.

### 5. Supplementary analyses

```bash
python experiments/supplementary/run_supplementary_experiments.py --cpu
```

These analyses are written separately under `results/supplementary` and do not
overwrite the primary experiment outputs.

## Output policy

Generated results, checkpoints, raw data, caches, and local environment files
are excluded by `.gitignore`. This keeps the public artifact focused on source
code and prevents accidental release of large or restricted data files.

## Research-integrity notes

- The fourth modality is real EU MRV fuel/emissions data, not cargo-manifest text.
- The study is an offline federated-learning simulation, not a claim of a live
  multi-institution deployment or formal differential-privacy protection.
- Supplementary analyses are stored separately from the primary experiments.
