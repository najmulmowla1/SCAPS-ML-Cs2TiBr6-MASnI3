# SCAPS-1D and explainable machine learning for Cs₂TiBr₆/MASnI₃ dual-absorber solar cells

Source-code release accompanying the manuscript:

**“Explainable Machine Learning Assisted SCAPS 1D Optimization of Lead Free Cs2TiBr6/MASnI3 Dual Absorber Perovskite Solar Cells”**

This repository provides the **available Python analysis workflow** from the supplied
`SCAPS_ML(4).ipynb` notebook, organized into 10 sequential scripts with a command-line
runner. It trains multi-output and PCE-only regression models, ranks simulated
configurations, produces feature-importance/SHAP analyses, exports figures, and writes
results tables. It does **not** rerun the SCAPS-1D numerical simulator.

## Data and availability

- **Simulation dataset (18,225 device configurations):** [Mendeley Data, V1](https://doi.org/10.17632/4fgnm4wftb.1).
- Original simulation framework: **SCAPS-1D v3.3.11** (not included; third-party simulator).
- Original notebook source: `notebooks/SCAPS_ML_original_source.ipynb` (cell sources retained; saved cell outputs removed).
- **Dataset is not bundled** in this GitHub package. Download it separately and supply its file path.

## Quick start

Tested for **syntax and a small synthetic-data smoke run** during packaging. Full numerical
replication of the manuscript requires the deposited data and original computational environment;
**it has not been verified here.**

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_pipeline.py --dataset "path/to/DATASET INITIALIZATION(1).xlsx" --output-dir results
```

If the Mendeley Data archive provides a `.csv` file instead, pass that path.
The accepted table headers are documented in `DATA_DICTIONARY.md`.

For a lightweight installation sanity check, run:

```bash
python run_pipeline.py --dataset "path/to/DATASET INITIALIZATION(1).xlsx" --output-dir results_quick --quick-check --no-shap
```

**Important:** `--quick-check` fits only small Extra Trees models (12 estimators) and
uses two permutation repeats. Its outputs are **not** manuscript results. With normal
settings, the notebook uses seed 42, an 80:20 random split, 350 estimators for Extra Trees,
and 10 permutation repeats. XGBoost, CatBoost, SHAP, and full model comparisons can be time-consuming.

## Repository layout

```text
SCAPS_ML_GitHub_Release/
├── run_pipeline.py              # main command-line entry point
├── stages/
│   ├── 01_load_and_clean.py     # load, rename, validate, derive ETL/HTL
│   ├── 02_descriptive_analysis.py
│   ├── 03_descriptive_figures.py
│   ├── 04_ml_setup.py           # features, 80:20 split, preprocessing, model factories
│   ├── 05_train_models.py       # multi-output and PCE-only regressions
│   ├── 06_rank_configurations.py
│   ├── 07_explainability.py     # permutation importance and SHAP
│   ├── 08_ml_figures.py
│   ├── 09_research_questions.py
│   └── 10_export_results.py    # Excel, CSV, metadata, trained models
├── notebooks/SCAPS_ML_original_source.ipynb
├── data/README.md
├── tests/test_repository.py
├── DATA_DICTIONARY.md
├── SCOPE_AND_LIMITATIONS.md
├── requirements.txt
└── .gitignore
```

Stage files are **ordered sections of the original notebook**, not independent entry
points: use `python run_pipeline.py` rather than executing a stage directly.
The runner deliberately uses a shared namespace to preserve notebook dependencies.

## Output files

The runner writes `results/SCAPS_ML_Crucial_Results.xlsx`, `results/tables_csv/`,
`results/figures_500dpi/`, `results/trained_models/`, and `results/run_metadata.json`.
Excel results include the simulation dataset overview, grouped analyses, ranking,
predictions, model metrics, permutation/SHAP importance where available, and a
research-question summary.

## Scientific interpretation

The regression models learn **SCAPS-1D simulated responses**, not measured device
performance. The reported data are from a single-junction device with two adjacent
absorber layers, not a conventional tandem containing independent subcells.

The notebook selects the best model using the held-out test split; therefore reported
test errors should **not** be represented as an independently confirmed post-selection
estimate. The random split also places neighboring simulated parameter configurations
in training and testing. All-dataset predictions include in-sample observations.

## What this repository does not provide

The provided notebook does **not** contain the distinct scripts or full raw inputs for
all additional results in the revised manuscript, including: the five-fold/seed/band-gap
holdout validation exercises, the retrospective screening-budget benchmark, and the
independent SCAPS-1D parameter sensitivity sweeps (defects, mobility, electrical
resistance, work function, interfaces). See `SCOPE_AND_LIMITATIONS.md`.

Do not claim that this archive reproduces every manuscript table or figure until those
missing scripts and input files have been supplied and their outputs verified.

## Citation

If you use the archived simulation data, cite the Mendeley Data record:
https://doi.org/10.17632/4fgnm4wftb.1

Please also cite the associated journal article when its final bibliographic citation
and DOI become available.

## License

No source-code license was specified by the authors in the supplied materials.
Before public release, the authors should select and add a suitable open-source license
(e.g., MIT or BSD-3-Clause) if they intend to authorize modification and redistribution.
