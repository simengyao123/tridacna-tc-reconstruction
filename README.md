# Tridacna tropical cyclone disturbance reconstruction

Aligned with the manuscript revision dated 10 October 2026.

Code and data accompanying **A computational framework for identifying tropical cyclone-related disturbances in modern-to-fossil Tridacna daily growth records**.

This project calculates four local TC Disturbance Index (TDI) descriptors, evaluates modern-specimen BiLSTM-Attention classifiers, compares modern and fossil feature distributions, explains TC-class predictions, and reconstructs potential disturbance events in fossil specimen A114.

The supplied trained models reproduce the manuscript's modern classification results and its 81-event A114 reconstruction. This is a local, organized submission candidate. A public repository URL, author-approved license, data reuse terms and citation metadata remain to be supplied before submission. See [scientific and implementation notes](docs/SCIENTIFIC_NOTES.md).

## Installation

Use Python 3.11 in a dedicated environment. Validation used Python 3.11.15 on Windows with the versions in `requirements.txt`. The PyTorch build used during validation was 2.6.0+cu124. A CPU installation is sufficient for inference; training and SHAP can use a compatible GPU automatically.

```console
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

On Linux/macOS, activate with `source .venv/bin/activate`. Run all commands below from this project's directory. A clean installation on another operating system has not been tested. Installation needs internet access; the supplied-data workflow itself does not download anything.

## Reproduce the main results with supplied checkpoints

```console
python run.py reproduce
```

This runs data preparation, modern-checkpoint verification, A114 inference, event extraction, domain comparison and feature plots. It uses `models/reference` and writes to `results/default`. It does not train a replacement model. Expected results:

| Result | Expected value |
| --- | --- |
| XB10 records; held-out records | 587; 177 |
| LHJ2 June–November records; held-out records | 1098; 330 |
| XB10 overall accuracy | 0.8531073446 (0.85 in the manuscript) |
| LHJ2 overall accuracy | 0.8939393939 (0.89 in the manuscript) |
| XB10 TC F1 | 0.8243243243 |
| LHJ2 TC F1 | 0.5882352941 |
| A114 daily records | 4383 |
| A114 candidate segments; final events | 60; 81 |
| A114 model events in June–November | 50/81, 61.7% |
| Manual reference events in June–November | 58/96, 60.4% |
| Mean standardized Wasserstein, LHJ2 / XB10 | 0.286028 / 1.211292 |
| MMD squared, LHJ2 / XB10 | 0.063717 / 0.603150 |

The checkpoint evaluation requires agreement with archived classifications and maximum probability difference below `1e-5`. It fails visibly if the comparison differs. These outputs verify execution and agreement with the saved experiment; they do not resolve the evaluation-design limitations described below.

## Other analyses

After `prepare` or `reproduce`:

```console
python run.py rf --sample XB10
python run.py rf --sample LHJ2
python run.py shap --sample XB10
python run.py shap --sample LHJ2
python -m unittest discover -s tests -v
```

SHAP uses all held-out TC-labelled days, 50 training-background samples, 200 Kernel SHAP samples and 1,000 episode bootstrap replicates. Raw attribution arrays, additivity checks, confidence intervals and figures are saved. Random Forest uses 300 trees, balanced class weights and flattened 9-by-6 windows. Its worker count defaults to one for compatibility with restricted Windows environments; set `rf_n_jobs` in the JSON config to change it.

## Training and ablation experiments

```console
python run.py prepare --run-dir results/retrained
python run.py train --sample XB10 --run-dir results/retrained
python run.py train --sample LHJ2 --run-dir results/retrained
python run.py predict --run-dir results/retrained --model-dir results/retrained/train
python run.py events --run-dir results/retrained
```

Training writes new models and scalers under the selected run directory's `train` folder. Reference models are separate. Set `--model-dir` explicitly to use newly trained models in later analyses. Reusing a run directory and sample overwrites that stage's outputs; use distinct directories for experiments.

Each modern sequence is split chronologically: the first 70% is used for parameter optimization, and TC-class F1 on the remaining 30% is used for early stopping, checkpoint selection and reported evaluation.

The supplied configurations `basic_features.json`, `window_3_3.json`, and `window_3_5.json` expose the feature and temporal-window experiments in Tables 4–5. For example:

```console
python run.py prepare --run-dir results/basic --config configs/basic_features.json
python run.py train --sample LHJ2 --run-dir results/basic --config configs/basic_features.json
```

These alternative experiments require retraining; do not pair their changed input shapes with the six-feature reference checkpoints.

## Files and options

| Path | Contents |
| --- | --- |
| `data/raw` | Supplied daily DGR, DGRV and label workbooks, copied unchanged |
| `data/reference_features` | Supplied precomputed features used for numerical comparison |
| `models/reference` | Supplied BiLSTM weights and fitted standardization objects |
| `reference_results` | Supplied modern classification, training-history and SHAP result files |
| `scripts` | Named analysis stages with English comments and docstrings |
| `configs` | Main settings and separately named experiment variants |
| `tests` | Calendar-gap, TDI-window and event-boundary regression checks |
| `docs` | Data dictionary, environment notes, manuscript mapping and verification record |
| `results/default` | Created when running the workflow; generated outputs are not bundled |

`python run.py --help` lists stages. Every stage accepts `--config`, `--run-dir`, and `--model-dir`; sample-specific stages use `--sample XB10` or `--sample LHJ2`. Relative paths are resolved from the current working directory. The internal data paths resolve from the project root. Figures are generated without interactive windows.

Each run saves a timestamped manifest containing the configuration, package versions, platform and completion status, plus per-stage logs. `docs/source_manifest.json` records SHA-256 hashes for bundled data, reference models and reference results, without local machine paths.

## Interpretation and citation

A114 labels are manual growth-anomaly reference markers, not meteorological ground truth. Its dates from 2024 to 2035 are a virtual indexing calendar, not actual fossil calendar years. The event-stage output uses relative years. The model estimates potential TC-related disturbances retrospectively, using both earlier and later growth information.

See [data dictionary](docs/DATA_DICTIONARY.md), [manuscript mapping](docs/MANUSCRIPT_MAPPING.md), and [verification record](docs/VERIFICATION.md). Citation and licensing details are intentionally pending author decisions; this directory does not grant an open-source or data redistribution license.
