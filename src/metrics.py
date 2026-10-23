import numpy as np
import pandas as pd


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Calculating mean absolute error between actuals and predictions.
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Calculating root mean squared error penalizing large forecast errors.
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Calculating weighted absolute percentage error robust to zero-demand instances.
    total_actual = np.sum(y_true)
    if total_actual == 0:
        return 0.0
    return float((np.sum(np.abs(y_true - y_pred)) / total_actual) * 100)


def calculate_mape_nonzero(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Calculating mean absolute percentage error exclusively on non-zero actuals.
    mask = y_true > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def calculate_forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Calculating percentage forecast bias indicating systematic over or under prediction.
    total_actual = np.sum(y_true)
    if total_actual == 0:
        return 0.0
    return float((np.sum(y_pred - y_true) / total_actual) * 100)


def evaluate_forecasts(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model"
) -> dict:
    # Evaluating complete suite of demand forecasting accuracy metrics.
    # Ensuring forecasts do not contain negative values.
    y_pred_clipped = np.clip(y_pred, 0, None)

    return {
        "model": model_name,
        "mae": round(calculate_mae(y_true, y_pred_clipped), 3),
        "rmse": round(calculate_rmse(y_true, y_pred_clipped), 3),
        "wape_pct": round(calculate_wape(y_true, y_pred_clipped), 2),
        "mape_nonzero_pct": round(
            calculate_mape_nonzero(y_true, y_pred_clipped), 2
        ),
        "bias_pct": round(calculate_forecast_bias(y_true, y_pred_clipped), 2),
    }
