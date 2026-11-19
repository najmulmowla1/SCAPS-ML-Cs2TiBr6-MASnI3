
# ============================================================
# 11. Feature importance and explainability
# ============================================================

# Fit preprocessor from best PCE model and recover transformed feature names
best_pce_model_preprocessor = best_pce_model.named_steps["preprocessor"]
feature_names_transformed = get_feature_names_from_preprocessor(best_pce_model_preprocessor)

# Tree/model-based feature importance when available
model_based_importance = pd.DataFrame()
final_estimator = best_pce_model.named_steps["model"]
if hasattr(final_estimator, "feature_importances_"):
    model_based_importance = pd.DataFrame({
        "Feature": feature_names_transformed,
        "Importance": final_estimator.feature_importances_,
        "Method": f"{best_pce_model_name} built-in feature importance"
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

# Permutation importance on the full pipeline using test set
print("Computing permutation importance for PCE. This may take a few minutes.")
perm = permutation_importance(
    best_pce_model, X_test_pce, y_test_pce,
    n_repeats=2 if QUICK_CHECK else 10, random_state=SEED, scoring="r2", n_jobs=2 if QUICK_CHECK else -1
)
permutation_importance_df = pd.DataFrame({
    "Feature": ml_features,
    "Importance_Mean_R2_Decrease": perm.importances_mean,
    "Importance_Std": perm.importances_std,
    "Method": "Permutation importance on PCE R2"
}).sort_values("Importance_Mean_R2_Decrease", ascending=False).reset_index(drop=True)

# SHAP analysis with fallback
shap_global_df = pd.DataFrame()
shap_success = False
try:
    if NO_SHAP:
        raise ImportError("SHAP calculation disabled by --no-shap")
    import shap
    # Transform data because most estimators expect encoded features
    X_train_transformed = best_pce_model_preprocessor.transform(X_train_pce)
    X_test_transformed = best_pce_model_preprocessor.transform(X_test_pce)
    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    # Use a sample to keep the notebook fast and stable
    sample_size = min(1000, X_test_transformed.shape[0])
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(X_test_transformed.shape[0], size=sample_size, replace=False)
    X_shap = X_test_transformed[sample_idx]

    # Prefer TreeExplainer for tree models. Use generic Explainer as fallback.
    try:
        explainer = shap.TreeExplainer(final_estimator)
        shap_values = explainer.shap_values(X_shap)
    except Exception:
        background_size = min(200, X_train_transformed.shape[0])
        background_idx = rng.choice(X_train_transformed.shape[0], size=background_size, replace=False)
        explainer = shap.Explainer(final_estimator.predict, X_train_transformed[background_idx])
        shap_values = explainer(X_shap).values

    if isinstance(shap_values, list):
        shap_values_arr = shap_values[0]
    else:
        shap_values_arr = np.asarray(shap_values)

    mean_abs_shap = np.mean(np.abs(shap_values_arr), axis=0)
    shap_global_df = pd.DataFrame({
        "Feature": feature_names_transformed,
        "Mean_Absolute_SHAP": mean_abs_shap,
        "Method": f"SHAP based on {best_pce_model_name}"
    }).sort_values("Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)
    shap_success = True

    # SHAP bar figure
    plt.figure(figsize=(7.2, 5.0))
    shap_plot_df = shap_global_df.head(15).iloc[::-1]
    plt.barh(shap_plot_df["Feature"], shap_plot_df["Mean_Absolute_SHAP"])
    plt.xlabel("Mean absolute SHAP value")
    plt.title("Global SHAP importance for PCE prediction")
    save_fig("Figure_12_SHAP_Global_Importance_PCE.jpg")

except Exception as exc:
    print("SHAP was skipped or could not be completed; permutation importance is retained.")
    print("Details:", exc)

print("Permutation importance:")
display(permutation_importance_df)

if len(model_based_importance) > 0:
    print("Model-based feature importance:")
    display(model_based_importance.head(20))

if shap_success:
    print("SHAP global importance:")
    display(shap_global_df.head(20))
