# Release scope and verification limitations

## Included directly from the uploaded original notebook

- Dataset import, column normalization, transport-layer extraction, cleaning.
- Grouped analysis for ETL/HTL pairs and six absorber design parameters.
- Original descriptive plots and 500-dpi figure exports.
- Train/test split (80:20, seed 42), pipeline-fitted imputation, encoding and scaling.
- Regression model definitions, multi-output model evaluation and PCE-only model evaluation.
- Full-grid prediction, SCAPS and ML ranking, consensus rank-sum selection.
- Permutation importance, optional SHAP, comparison plots and spreadsheet/CSV/model exports.

## Not present in the supplied notebook, therefore not independently reproduced here

1. The additional five-fold cross-validation, five random-seed partition analysis, and
   holdout of individual MASnI3 band-gap levels used for the manuscript's Table 9.
2. The retrospective ML-versus-random evaluation-budget screening experiments (20
   verification candidates per budget, 10 repetitions) used for the manuscript's Figure 7.
3. The independent SCAPS-1D defect-density, carrier-mobility, resistance, contact
   work-function, and band-offset sensitivity sweeps used outside the 18,225-row ML dataset.
4. SCAPS-1D input definitions, simulator executable or simulation automation needed to
   regenerate the 18,225 rows from device physics assumptions.
5. Source files for the manuscript's final typeset figures; notebook figures may differ
   from journal figure numbering/layout.

The manuscript reports these additional results but the upload did not contain their
underlying scripts. Publishing the provided code is beneficial, but **full article-level
computational reproducibility requires adding the missing code and inputs**.

## Portability changes relative to the uploaded notebook

- Replaced its fixed `/content/...` Colab paths with explicit `--dataset`/`--output-dir`.
- Removed automatic notebook pip installation and Colab-only downloads; dependencies
  are supplied in `requirements.txt`.
- Added CSV input, headless plotting and a CLI `display` helper.
- Added checks for invalid (nonpositive) doping/acceptor densities and unrecognized
  transport-layer names; warn when input contains a row count other than 18,225.
- Kept the original model definitions, split/seed, ranking, and most figure/export code.
- Added a separate **non-publication** quick-check mode to exercise the end-to-end pipeline.
- Stripped notebook outputs from the included archival notebook; cell source retained.

## Publication/verification note

This package was syntax-checked and smoke-tested on synthetic **test-only** data. The
published dataset itself was not mounted during packaging, so the claimed numerical
results have not been reexecuted/verified. The test dataset is not part of the release.

Some dependencies are version-dependent; requirements use compatibility ranges rather
than a fully frozen environment. Reproduce with a documented Python environment and
record installed versions if exact numerical replication is required.

- Added a CatBoost estimator-tag adapter for recent scikit-learn versions. It changes estimator metadata only, not CatBoost model hyperparameters.
