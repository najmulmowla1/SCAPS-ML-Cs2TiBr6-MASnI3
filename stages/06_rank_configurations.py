# ============================================================
# 10. Predict full dataset and rank optimal configurations
# ============================================================

# Predict all four targets using best multi-output model
full_multi_pred = np.asarray(best_multi_model.predict(X))

full_predictions = df.copy()

for i, target in enumerate(target_cols):
    full_predictions[f"ML_Predicted_{target}"] = full_multi_pred[:, i]
    full_predictions[f"ML_Error_{target}"] = (
        full_predictions[target] - full_predictions[f"ML_Predicted_{target}"]
    )
    full_predictions[f"ML_Absolute_Error_{target}"] = np.abs(
        full_predictions[f"ML_Error_{target}"]
    )

# Predict PCE using PCE-only best model
full_predictions["ML_PCE_Only_Predicted_PCE"] = best_pce_model.predict(X)

full_predictions["ML_PCE_Only_Error"] = (
    full_predictions["PCE"] - full_predictions["ML_PCE_Only_Predicted_PCE"]
)

full_predictions["ML_PCE_Only_Absolute_Error"] = np.abs(
    full_predictions["ML_PCE_Only_Error"]
)

# ------------------------------------------------------------
# Add ranking columns BEFORE creating ranked subsets
# ------------------------------------------------------------

full_predictions["SCAPS_PCE_Rank"] = full_predictions["PCE"].rank(
    ascending=False,
    method="min"
)

full_predictions["ML_PCE_Rank"] = full_predictions["ML_PCE_Only_Predicted_PCE"].rank(
    ascending=False,
    method="min"
)

full_predictions["Consensus_Rank_Score"] = (
    full_predictions["SCAPS_PCE_Rank"] + full_predictions["ML_PCE_Rank"]
)

# Rank by simulated SCAPS PCE
ranked_by_scaps_pce = (
    full_predictions
    .sort_values("PCE", ascending=False)
    .head(100)
    .reset_index(drop=True)
)

# Rank by ML-predicted PCE
ranked_by_ml_pce = (
    full_predictions
    .sort_values("ML_PCE_Only_Predicted_PCE", ascending=False)
    .head(100)
    .reset_index(drop=True)
)

# Consensus high-performance candidates
consensus_top_100 = (
    full_predictions
    .sort_values("Consensus_Rank_Score", ascending=True)
    .head(100)
    .reset_index(drop=True)
)

# ------------------------------------------------------------
# Important output columns
# ------------------------------------------------------------

key_output_cols = [
    "configuration", "ETL", "HTL",
    "top_absorber_thickness_um", "bottom_absorber_thickness_um",
    "top_absorber_bandgap_ev", "bottom_absorber_bandgap_ev",
    "top_absorber_doping_cm3", "bottom_absorber_acceptor_cm3",
    "Voc", "Jsc", "FF", "PCE",
    "ML_PCE_Only_Predicted_PCE", "ML_PCE_Only_Absolute_Error",
    "SCAPS_PCE_Rank", "ML_PCE_Rank", "Consensus_Rank_Score"
]

# Keep only columns that exist, to avoid future column-name errors
available_key_output_cols = [
    col for col in key_output_cols if col in full_predictions.columns
]

missing_cols = [
    col for col in key_output_cols if col not in full_predictions.columns
]

if missing_cols:
    print("Warning: These expected columns were not found:")
    print(missing_cols)

print("Top 20 configurations ranked by SCAPS PCE:")
display(ranked_by_scaps_pce[available_key_output_cols].head(20))

print("Top 20 configurations ranked by ML-predicted PCE:")
display(ranked_by_ml_pce[available_key_output_cols].head(20))

print("Top 20 consensus high-performance configurations:")
display(consensus_top_100[available_key_output_cols].head(20))
