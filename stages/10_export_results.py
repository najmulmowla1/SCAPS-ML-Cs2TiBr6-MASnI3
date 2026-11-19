# ============================================================
# 14. Export all crucial tables
# ============================================================

import json
import joblib
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

excel_path = OUTPUT_DIR / "SCAPS_ML_Crucial_Results.xlsx"

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def make_unique_columns(df_in):
    """
    Make DataFrame column names unique.
    This prevents pandas from returning a DataFrame when selecting one column.
    """
    df_out = df_in.copy()
    new_cols = []
    counts = {}

    for col in df_out.columns:
        col = str(col)
        if col not in counts:
            counts[col] = 0
            new_cols.append(col)
        else:
            counts[col] += 1
            new_cols.append(f"{col}_{counts[col]}")

    df_out.columns = new_cols
    return df_out


def select_existing_unique_columns(df_in, desired_cols):
    """
    Select columns that exist in df_in and remove duplicated requests.
    """
    selected = []
    seen = set()

    for col in desired_cols:
        if col in df_in.columns and col not in seen:
            selected.append(col)
            seen.add(col)

    return df_in[selected].copy()


def ensure_dataframe(obj, sheet_name="Table"):
    """
    Convert object to a valid DataFrame for Excel export.
    """
    if isinstance(obj, pd.DataFrame):
        df_obj = obj.copy()
    elif isinstance(obj, pd.Series):
        df_obj = obj.reset_index()
    elif isinstance(obj, list):
        df_obj = pd.DataFrame(obj)
    elif isinstance(obj, dict):
        df_obj = pd.DataFrame([obj])
    else:
        df_obj = pd.DataFrame({"Note": [f"{sheet_name} could not be converted to a DataFrame."]})

    if df_obj.shape[1] == 0:
        df_obj = pd.DataFrame({"Note": [f"{sheet_name} is empty."]})

    df_obj = make_unique_columns(df_obj)
    return df_obj


def safe_excel_sheet_name(name, used_names):
    """
    Excel sheet names must be <=31 characters and cannot contain some characters.
    This also prevents duplicated sheet names after truncation.
    """
    invalid_chars = ["\\", "/", "*", "?", ":", "[", "]"]
    safe = str(name)

    for ch in invalid_chars:
        safe = safe.replace(ch, "_")

    safe = safe[:31]

    if safe == "":
        safe = "Sheet"

    base = safe
    counter = 1

    while safe in used_names:
        suffix = f"_{counter}"
        safe = base[:31 - len(suffix)] + suffix
        counter += 1

    used_names.add(safe)
    return safe


# ------------------------------------------------------------
# Prepare file inventory
# ------------------------------------------------------------

files_created = []

for path in sorted(OUTPUT_DIR.rglob("*")):
    if path.is_file():
        files_created.append({
            "File": path.name,
            "Path": str(path),
            "Size_kB": round(path.stat().st_size / 1024, 2)
        })

files_created_df = pd.DataFrame(files_created)

# Ensure figure table is updated
figure_records_df = pd.DataFrame(figure_records) if "figure_records" in globals() else pd.DataFrame(
    {"Note": ["No figure records were found."]}
)

# ------------------------------------------------------------
# Prepare safe output-column lists
# ------------------------------------------------------------

# Avoid duplicated ML columns
ml_cols = [c for c in full_predictions.columns if str(c).startswith("ML_")]

full_prediction_output_cols = []
seen_cols = set()

for col in key_output_cols + ml_cols:
    if col in full_predictions.columns and col not in seen_cols:
        full_prediction_output_cols.append(col)
        seen_cols.add(col)

available_key_output_cols = [
    col for col in key_output_cols if col in full_predictions.columns
]

best_overall_cols = [
    col for col in key_output_cols[:13] if col in best_overall.columns
]

# ------------------------------------------------------------
# Build sheet dictionary safely
# ------------------------------------------------------------

sheets = {
    "README": pd.DataFrame([
        {
            "Item": "Study",
            "Description": "Explainable ML-assisted SCAPS-1D optimization of Pb-free Cs2TiBr6/MASnI3 dual-absorber PSC"
        },
        {
            "Item": "Device",
            "Description": "FTO/ETL/Cs2TiBr6/MASnI3/HTL/Au"
        },
        {
            "Item": "Dataset",
            "Description": str(DATASET_PATH)
        },
        {
            "Item": "Rows after cleaning",
            "Description": len(df)
        },
        {
            "Item": "Best PCE model",
            "Description": best_pce_model_name
        },
        {
            "Item": "Best multi-output model",
            "Description": best_multi_model_name
        },
        {
            "Item": "Generated at",
            "Description": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
    ]),

    "Dataset_Summary": dataset_summary,
    "Input_Parameter_Ranges": input_parameter_ranges,
    "Output_Statistics": output_statistics,
    "Cleaned_Dataset": df,

    "Best_Overall_Config": best_overall[best_overall_cols],
    "Top_50_SCAPS_PCE": ranked_by_scaps_pce[available_key_output_cols].head(50),
    "Top_100_SCAPS_PCE": ranked_by_scaps_pce[available_key_output_cols],
    "Top_100_ML_Predicted_PCE": ranked_by_ml_pce[available_key_output_cols],
    "Consensus_Top_100": consensus_top_100[available_key_output_cols],

    "ETL_Ranking": etl_ranking,
    "HTL_Ranking": htl_ranking,
    "ETL_HTL_Ranking": etl_htl_ranking,
    "Configuration_Ranking": configuration_ranking,

    "Top_Thickness_Effect": top_thickness_effect,
    "Bottom_Thickness_Effect": bottom_thickness_effect,
    "Top_Bandgap_Effect": top_bandgap_effect,
    "Bottom_Bandgap_Effect": bottom_bandgap_effect,
    "Top_Doping_Effect": top_doping_effect,
    "Bottom_Acceptor_Effect": bottom_acceptor_effect,

    "Thickness_Interaction": thickness_interaction,
    "Bandgap_Interaction": bandgap_interaction,
    "Doping_Interaction": doping_interaction,

    "Model_Metrics_All_Targets": multi_model_metrics,
    "Model_Metrics_PCE_Only": pce_model_metrics,
    "Predictions_Best_PCE_Model": best_pce_predictions,
    "All_PCE_Model_Predictions": all_pce_predictions,
    "All_Multi_Output_Predictions": all_multi_predictions,

    "Full_Dataset_ML_Predictions": full_predictions[full_prediction_output_cols],

    "Feature_Importance_Permut": permutation_importance_df,

    "Feature_Importance_Model": (
        model_based_importance
        if isinstance(model_based_importance, pd.DataFrame) and len(model_based_importance) > 0
        else pd.DataFrame({"Note": ["Built-in model importance unavailable for the selected model."]})
    ),

    "SHAP_Global_PCE": (
        shap_global_df
        if isinstance(shap_global_df, pd.DataFrame) and len(shap_global_df) > 0
        else pd.DataFrame({"Note": ["SHAP was not completed. Use permutation importance instead."]})
    ),

    "Research_Q_Answers": research_question_answers,
    "Figures_500dpi": figure_records_df,
    "Files_Created": files_created_df,
}

# ------------------------------------------------------------
# Export to Excel with formatting
# ------------------------------------------------------------

used_sheet_names = set()

with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
    workbook = writer.book

    header_format = workbook.add_format({
        "bold": True,
        "bg_color": "#D9EAF7",
        "border": 1
    })

    number_format = workbook.add_format({
        "num_format": "0.000000"
    })

    text_wrap = workbook.add_format({
        "text_wrap": True,
        "valign": "top"
    })

    for sheet_name, table in sheets.items():
        safe_name = safe_excel_sheet_name(sheet_name, used_sheet_names)

        table_to_write = ensure_dataframe(table, sheet_name)

        # Excel limit: 1,048,576 rows total.
        # Keep 1 row for header.
        if len(table_to_write) > 1048575:
            table_to_write = table_to_write.head(1048575)

        table_to_write.to_excel(writer, sheet_name=safe_name, index=False)

        worksheet = writer.sheets[safe_name]
        worksheet.freeze_panes(1, 0)

        n_rows, n_cols = table_to_write.shape

        if n_cols > 0:
            worksheet.autofilter(0, 0, max(n_rows, 1), n_cols - 1)

            for col_idx, col_name in enumerate(table_to_write.columns):
                worksheet.write(0, col_idx, col_name, header_format)

                series = table_to_write.iloc[:, col_idx]
                width = min(max(len(str(col_name)) + 2, 12), 35)

                if pd.api.types.is_object_dtype(series):
                    sample_lens = (
                        series
                        .dropna()
                        .astype(str)
                        .head(100)
                        .map(len)
                    )

                    if len(sample_lens) > 0:
                        width = min(max(width, int(sample_lens.quantile(0.90)) + 2), 45)

                    worksheet.set_column(col_idx, col_idx, width, text_wrap)

                else:
                    worksheet.set_column(col_idx, col_idx, width, number_format)

# ------------------------------------------------------------
# Export selected CSV tables as well
# ------------------------------------------------------------

for name, table in sheets.items():
    if name in ["Cleaned_Dataset", "Full_Dataset_ML_Predictions"]:
        continue

    try:
        table_to_csv = ensure_dataframe(table, name)
        table_to_csv.to_csv(TABLE_DIR / f"{name}.csv", index=False)
    except Exception as e:
        print(f"CSV export skipped for {name}: {e}")

# ------------------------------------------------------------
# Save models and metadata
# ------------------------------------------------------------

joblib.dump(
    best_pce_model,
    MODEL_DIR / f"best_pce_model_{best_pce_model_name.replace(' ', '_')}.joblib"
)

joblib.dump(
    best_multi_model,
    MODEL_DIR / f"best_multi_output_model_{best_multi_model_name.replace(' ', '_')}.joblib"
)

metadata = {
    "dataset_path": str(DATASET_PATH),
    "output_dir": str(OUTPUT_DIR),
    "clean_rows": int(len(df)),
    "best_pce_model": best_pce_model_name,
    "best_multi_output_model": best_multi_model_name,
    "best_scaps_configuration": {
        "configuration": str(best_config["configuration"]),
        "ETL": str(best_config["ETL"]),
        "HTL": str(best_config["HTL"]),
        "PCE": float(best_config["PCE"]),
        "Voc": float(best_config["Voc"]),
        "Jsc": float(best_config["Jsc"]),
        "FF": float(best_config["FF"]),
    },
    "created_at": datetime.now().isoformat(),
}

with open(OUTPUT_DIR / "run_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Saved main Excel result file:", excel_path)
print("Saved figures to:", FIG_DIR)
print("Saved models to:", MODEL_DIR)
print("Saved CSV tables to:", TABLE_DIR)
