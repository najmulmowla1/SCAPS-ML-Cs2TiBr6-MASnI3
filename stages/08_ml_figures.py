
# ============================================================
# 12. Figures for ML performance and feature importance
# ============================================================

# Best PCE model predictions
best_pred = best_pce_predictions.copy()

# Figure 10: Actual vs predicted PCE
plt.figure(figsize=(5.6, 5.0))
plt.scatter(best_pred["Actual_PCE"], best_pred["Predicted_PCE"], alpha=0.65, s=18)
min_v = min(best_pred["Actual_PCE"].min(), best_pred["Predicted_PCE"].min())
max_v = max(best_pred["Actual_PCE"].max(), best_pred["Predicted_PCE"].max())
plt.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=1.0)
plt.xlabel("SCAPS-1D PCE (%)")
plt.ylabel("ML-predicted PCE (%)")
plt.title(f"Actual vs predicted PCE ({best_pce_model_name})")
save_fig("Figure_10_Actual_vs_Predicted_PCE.jpg")

# Figure 11: Residuals
plt.figure(figsize=(6.2, 4.2))
residuals = best_pred["Actual_PCE"] - best_pred["Predicted_PCE"]
plt.scatter(best_pred["Predicted_PCE"], residuals, alpha=0.65, s=18)
plt.axhline(0, linestyle="--", linewidth=1.0)
plt.xlabel("ML-predicted PCE (%)")
plt.ylabel("Residual: SCAPS PCE - predicted PCE (%)")
plt.title(f"PCE residual analysis ({best_pce_model_name})")
save_fig("Figure_11_PCE_Residual_Analysis.jpg")

# Figure 13: Permutation importance
plt.figure(figsize=(7.2, 4.8))
plot_perm = permutation_importance_df.head(12).iloc[::-1]
plt.barh(plot_perm["Feature"], plot_perm["Importance_Mean_R2_Decrease"])
plt.xlabel("Mean decrease in R2")
plt.title("Permutation feature importance for PCE")
save_fig("Figure_13_Permutation_Importance_PCE.jpg")

# Figure 14: model comparison based on PCE RMSE
plt.figure(figsize=(7.0, 4.4))
plot_metrics = pce_model_metrics.sort_values("RMSE", ascending=True)
plt.barh(plot_metrics["Model"].iloc[::-1], plot_metrics["RMSE"].iloc[::-1])
plt.xlabel("RMSE for PCE prediction")
plt.title("PCE prediction model comparison")
save_fig("Figure_14_Model_Comparison_PCE_RMSE.jpg")

figure_records_df = pd.DataFrame(figure_records)
display(figure_records_df)
