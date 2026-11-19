# Compatibility bridge: CatBoost 1.2.x can be fitted but cannot be predicted
# inside a scikit-learn >=1.6 Pipeline without modern estimator tags.
# This adapter changes metadata only; the CatBoost algorithm and parameters
# are otherwise unchanged from the source notebook.
try:
    from catboost import CatBoostRegressor as _NativeCatBoostRegressor
    from sklearn.base import BaseEstimator

    class CompatibleCatBoostRegressor(_NativeCatBoostRegressor):
        def __sklearn_tags__(self):
            tags = BaseEstimator.__sklearn_tags__(self)
            tags.estimator_type = "regressor"
            return tags
except ImportError:
    CompatibleCatBoostRegressor = None


# ============================================================
# 6. Prepare ML features and targets
# ============================================================

categorical_features = ["ETL", "HTL"]
ml_numeric_features = [
    "top_absorber_thickness_um",
    "bottom_absorber_thickness_um",
    "top_absorber_bandgap_ev",
    "bottom_absorber_bandgap_ev",
    "log10_top_absorber_doping",
    "log10_bottom_absorber_acceptor",
]

ml_features = categorical_features + ml_numeric_features
X = df[ml_features].copy()
y = df[target_cols].copy()
y_pce = df["PCE"].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
X_train_pce, X_test_pce, y_train_pce, y_test_pce = train_test_split(X, y_pce, test_size=0.20, random_state=SEED)

try:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", onehot),
        ]), categorical_features),
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), ml_numeric_features),
    ],
    remainder="drop"
)

print("Feature columns:", ml_features)
print("Target columns:", target_cols)
print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])



# ============================================================
# 7. Model training utilities
# ============================================================

def get_available_models():
    models = {}
    models["Linear Regression"] = LinearRegression()
    models["Ridge Regression"] = Ridge(alpha=1.0, random_state=SEED)
    models["Random Forest"] = RandomForestRegressor(
        n_estimators=350, random_state=SEED, n_jobs=-1, max_depth=None, min_samples_leaf=1
    )
    models["Extra Trees"] = ExtraTreesRegressor(
        n_estimators=350, random_state=SEED, n_jobs=-1, max_depth=None, min_samples_leaf=1
    )
    models["Hist Gradient Boosting"] = MultiOutputRegressor(
        HistGradientBoostingRegressor(random_state=SEED, max_iter=300, learning_rate=0.06)
    )
    models["SVR"] = MultiOutputRegressor(SVR(C=20, epsilon=0.02, kernel="rbf"))
    models["MLP"] = MLPRegressor(
        hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
        max_iter=700, random_state=SEED, early_stopping=True, validation_fraction=0.1
    )

    # Optional XGBoost
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=500, learning_rate=0.04, max_depth=6,
                subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror",
                random_state=SEED, n_jobs=-1, reg_lambda=1.0
            )
        )
    except Exception as exc:
        print("XGBoost unavailable:", exc)

    # Optional CatBoost
    try:
        from catboost import CatBoostRegressor
        CatBoostRegressor = CompatibleCatBoostRegressor
        models["CatBoost"] = CatBoostRegressor(
            iterations=700, learning_rate=0.04, depth=7, loss_function="MultiRMSE",
            random_seed=SEED, verbose=False
        )
    except Exception as exc:
        print("CatBoost unavailable:", exc)

    return models


def get_available_pce_models():
    models = {}
    models["Linear Regression"] = LinearRegression()
    models["Ridge Regression"] = Ridge(alpha=1.0, random_state=SEED)
    models["Random Forest"] = RandomForestRegressor(
        n_estimators=350, random_state=SEED, n_jobs=-1, max_depth=None, min_samples_leaf=1
    )
    models["Extra Trees"] = ExtraTreesRegressor(
        n_estimators=350, random_state=SEED, n_jobs=-1, max_depth=None, min_samples_leaf=1
    )
    models["Gradient Boosting"] = GradientBoostingRegressor(random_state=SEED, n_estimators=350, learning_rate=0.04, max_depth=4)
    models["Hist Gradient Boosting"] = HistGradientBoostingRegressor(random_state=SEED, max_iter=350, learning_rate=0.05)
    models["SVR"] = SVR(C=20, epsilon=0.02, kernel="rbf")
    models["MLP"] = MLPRegressor(
        hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
        max_iter=700, random_state=SEED, early_stopping=True, validation_fraction=0.1
    )
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=600, learning_rate=0.04, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror",
            random_state=SEED, n_jobs=-1, reg_lambda=1.0
        )
    except Exception as exc:
        print("XGBoost unavailable:", exc)
    try:
        from catboost import CatBoostRegressor
        CatBoostRegressor = CompatibleCatBoostRegressor
        models["CatBoost"] = CatBoostRegressor(
            iterations=800, learning_rate=0.04, depth=7,
            random_seed=SEED, verbose=False, loss_function="RMSE"
        )
    except Exception as exc:
        print("CatBoost unavailable:", exc)
    return models


def regression_metrics(y_true, y_pred, target_name):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    eps = 1e-12
    mape = np.mean(np.abs((np.asarray(y_true) - np.asarray(y_pred)) / np.maximum(np.abs(np.asarray(y_true)), eps))) * 100
    return {"Target": target_name, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE_percent": mape}


def get_feature_names_from_preprocessor(fitted_preprocessor):
    names = []
    # Categorical names
    cat_pipe = fitted_preprocessor.named_transformers_["cat"]
    ohe = cat_pipe.named_steps["onehot"]
    try:
        cat_names = list(ohe.get_feature_names_out(categorical_features))
    except Exception:
        cat_names = list(ohe.get_feature_names(categorical_features))
    names.extend(cat_names)
    names.extend(ml_numeric_features)
    return names
