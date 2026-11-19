#!/usr/bin/env python3
"""Sequential runner for the staged source extracted from SCAPS_ML(4).ipynb.

The stages deliberately share one namespace to preserve the original notebook's
scientific analysis sequence. This is a script-oriented reproducibility release,
not a new independent implementation of SCAPS-1D simulations.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import glob
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless figure generation for Linux/CI/Colab
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import joblib

STAGES = [
    "01_load_and_clean.py", "02_descriptive_analysis.py", "03_descriptive_figures.py",
    "04_ml_setup.py", "05_train_models.py", "06_rank_configurations.py",
    "07_explainability.py", "08_ml_figures.py", "09_research_questions.py",
    "10_export_results.py",
]

def display(value):
    """Portable replacement for notebook-only display() for CLI output."""
    try:
        if isinstance(value, (pd.Series, pd.DataFrame)):
            print(value.head(8).to_string(index=False))
            if len(value) > 8:
                print(f"... {len(value) - 8} further rows")
        else:
            print(value)
    except Exception:
        print(type(value).__name__)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce available SCAPS-1D / ML notebook analysis")
    parser.add_argument("--dataset", required=True, type=Path,
                        help="Path to the original SCAPS dataset (.xlsx or .csv) from Mendeley Data")
    parser.add_argument("--output-dir", type=Path, default=Path("results"),
                        help="Directory for generated figures, tables, and trained models")
    parser.add_argument("--quick-check", action="store_true", help="Smoke test only; 12-tree Extra Trees models; NOT manuscript results")
    parser.add_argument("--no-shap", action="store_true", help="Skip optional SHAP calculation")
    args = parser.parse_args(argv)
    dataset_path = args.dataset.expanduser().resolve()
    if not dataset_path.is_file():
        parser.error(f"Dataset not found: {dataset_path}")
    if dataset_path.suffix.lower() not in {".xlsx", ".xls", ".csv"}:
        parser.error("The input must be Excel (.xlsx/.xls) or CSV (.csv)")
    output_dir = args.output_dir.expanduser().resolve()
    fig_dir = output_dir / "figures_500dpi"
    model_dir = output_dir / "trained_models"
    table_dir = output_dir / "tables_csv"
    for d in (output_dir, fig_dir, model_dir, table_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Stage scripts are the original notebook cell bodies, separated by analytic task.
    # Sharing a namespace intentionally reproduces the notebook's variable dependencies.
    context = dict(globals())
    context.update({
        "__name__": "__main__", "DATASET_PATH": dataset_path,
        "OUTPUT_DIR": output_dir, "FIG_DIR": fig_dir,
        "MODEL_DIR": model_dir, "TABLE_DIR": table_dir,
        "SEED": 42, "QUICK_CHECK": bool(args.quick_check),
        "NO_SHAP": bool(args.no_shap),
    })
    np.random.seed(42)
    print("Input:", dataset_path)
    print("Outputs:", output_dir)
    if args.quick_check:
        print("QUICK-CHECK MODE: results are not intended for manuscript replication")
    stage_root = Path(__file__).resolve().parent / "stages"
    for index, stage in enumerate(STAGES, 1):
        source_path = stage_root / stage
        print(f"\n==== Stage {index}/{len(STAGES)}: {stage} ====")
        source = source_path.read_text(encoding="utf-8")
        exec(compile(source, str(source_path), "exec"), context)
    print("Finished. Main workbook:", output_dir / "SCAPS_ML_Crucial_Results.xlsx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
