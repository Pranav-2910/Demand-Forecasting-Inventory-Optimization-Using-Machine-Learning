from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.metrics import evaluate_forecasts

# Defining standard feature names utilized across all machine learning models.
MODEL_FEATURE_COLS = [
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_21",
    "sales_lag_28",
    "sales_lag_35",
    "sales_lag_42",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_max_7",
    "rolling_mean_14",
    "rolling_std_14",
    "rolling_max_14",
    "rolling_mean_28",
    "rolling_std_28",
    "rolling_max_28",
    "day_of_week",
    "day_of_month",
    "month",
    "year",
    "is_weekend",
    "has_event",
    "event_type_code",
    "snap_CA",
    "sell_price",
    "price_diff_to_mean",
    "discount_ratio",
    "is_discounted",
]


def prepare_model_matrices(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list = MODEL_FEATURE_COLS,
    target_col: str = "sales",
) -> dict:
    # Extracting feature matrices and target arrays for train, validation, and test splits.
    return {
        "X_train": df_train[feature_cols].values,
        "y_train": df_train[target_col].values,
        "X_val": df_val[feature_cols].values,
        "y_val": df_val[target_col].values,
        "X_test": df_test[feature_cols].values,
        "y_test": df_test[target_col].values,
        "feature_names": feature_cols,
    }


def train_and_tune_models(data_dict: dict) -> dict:
    # Instantiating candidate machine learning forecasting regressors.
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
    }

    results = {}
    trained_models = {}

    X_train, y_train = data_dict["X_train"], data_dict["y_train"]
    X_val, y_val = data_dict["X_val"], data_dict["y_val"]
    X_test, y_test = data_dict["X_test"], data_dict["y_test"]

    for name, model in models.items():
        # Fitting forecasting model on chronological training partition.
        print(f"Training {name} regressor...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Generating out-of-sample predictions on validation and test partitions.
        preds_val = np.clip(model.predict(X_val), 0, None)
        preds_test = np.clip(model.predict(X_test), 0, None)

        # Evaluating accuracy metrics on validation and test partitions.
        metrics_val = evaluate_forecasts(
            y_val, preds_val, model_name=f"{name} [Validation]"
        )
        metrics_test = evaluate_forecasts(
            y_test, preds_test, model_name=f"{name} [Test]"
        )

        results[name] = {
            "model": model,
            "preds_val": preds_val,
            "preds_test": preds_test,
            "metrics_val": metrics_val,
            "metrics_test": metrics_test,
        }

    return results, trained_models


def select_best_model(results: dict, metric_key: str = "wape_pct") -> tuple:
    # Selecting winning model based on lowest validation error metric.
    best_name = None
    best_score = float("inf")

    for name, res in results.items():
        score = res["metrics_val"][metric_key]
        if score < best_score:
            best_score = score
            best_name = name

    return best_name, results[best_name]


def save_model_artifact(
    model, filepath: str = "models/demand_forecaster.joblib"
) -> None:
    # Persisting serialized trained model to disk.
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    print(f"Model saved successfully to {filepath}")
