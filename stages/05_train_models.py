
# ============================================================
# 8. Train and evaluate multi-output models
# ============================================================

multi_models = get_available_models()
if QUICK_CHECK:
    multi_models = {"Extra Trees": ExtraTreesRegressor(n_estimators=12, random_state=SEED, n_jobs=2)}
trained_multi_models = {}
multi_model_metric_rows = []
multi_predictions_frames = []

for model_name, model in multi_models.items():
    print(f"Training multi-output model: {model_name}")
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    pred = np.asarray(pred)
    trained_multi_models[model_name] = pipe

    for i, target in enumerate(target_cols):
        metric = regression_metrics(y_test[target].values, pred[:, i], target)
        metric["Model"] = model_name
        multi_model_metric_rows.append(metric)

    pred_df = X_test.copy().reset_index(drop=True)
    actual_df = y_test.reset_index(drop=True).add_prefix("Actual_")
    predicted_df = pd.DataFrame(pred, columns=[f"Predicted_{c}" for c in target_cols])
    temp = pd.concat([pred_df, actual_df, predicted_df], axis=1)
    temp["Model"] = model_name
    multi_predictions_frames.append(temp)

multi_model_metrics = pd.DataFrame(multi_model_metric_rows)
multi_model_metrics = multi_model_metrics[["Model", "Target", "RMSE", "MAE", "R2", "MAPE_percent"]]

# Select best multi-output model by PCE RMSE first, then R2
pce_metrics = multi_model_metrics[multi_model_metrics["Target"] == "PCE"].copy()
pce_metrics = pce_metrics.sort_values(["RMSE", "R2"], ascending=[True, False]).reset_index(drop=True)
best_multi_model_name = pce_metrics.loc[0, "Model"]
best_multi_model = trained_multi_models[best_multi_model_name]

print("Best multi-output model based on PCE RMSE:", best_multi_model_name)
display(multi_model_metrics.sort_values(["Target", "RMSE"]).reset_index(drop=True))

all_multi_predictions = pd.concat(multi_predictions_frames, ignore_index=True)



# ============================================================
# 9. Train and evaluate PCE-only models
# ============================================================

pce_models = get_available_pce_models()
if QUICK_CHECK:
    pce_models = {"Extra Trees": ExtraTreesRegressor(n_estimators=12, random_state=SEED, n_jobs=2)}
trained_pce_models = {}
pce_model_metric_rows = []
pce_predictions_frames = []

for model_name, model in pce_models.items():
    print(f"Training PCE-only model: {model_name}")
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train_pce, y_train_pce)
    pred = pipe.predict(X_test_pce)
    trained_pce_models[model_name] = pipe

    metric = regression_metrics(y_test_pce.values, pred, "PCE")
    metric["Model"] = model_name
    pce_model_metric_rows.append(metric)

    pred_df = X_test_pce.copy().reset_index(drop=True)
    pred_df["Actual_PCE"] = y_test_pce.reset_index(drop=True)
    pred_df["Predicted_PCE"] = pred
    pred_df["Absolute_Error"] = np.abs(pred_df["Actual_PCE"] - pred_df["Predicted_PCE"])
    pred_df["Model"] = model_name
    pce_predictions_frames.append(pred_df)

pce_model_metrics = pd.DataFrame(pce_model_metric_rows)
pce_model_metrics = pce_model_metrics[["Model", "Target", "RMSE", "MAE", "R2", "MAPE_percent"]]
pce_model_metrics = pce_model_metrics.sort_values(["RMSE", "R2"], ascending=[True, False]).reset_index(drop=True)
best_pce_model_name = pce_model_metrics.loc[0, "Model"]
best_pce_model = trained_pce_models[best_pce_model_name]

print("Best PCE-only model:", best_pce_model_name)
display(pce_model_metrics)

all_pce_predictions = pd.concat(pce_predictions_frames, ignore_index=True)
best_pce_predictions = all_pce_predictions[all_pce_predictions["Model"] == best_pce_model_name].copy()
