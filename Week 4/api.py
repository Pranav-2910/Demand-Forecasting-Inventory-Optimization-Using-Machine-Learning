from pathlib import Path
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from scipy.stats import norm

# Initializing FastAPI web application instance.
app = FastAPI(
    title="Demand Forecasting & Inventory Replenishment API",
    description="Enterprise REST API serving production machine learning demand forecasts and automated inventory replenishment recommendations.",
    version="1.0.0",
)

# Locating project root directory dynamically.
root_dir = Path(".").resolve()
if not (root_dir / "Week 3").exists() and (root_dir / ".." / "Week 3").exists():
    root_dir = (root_dir / "..").resolve()

forecast_path = root_dir / "Week 3" / "forecast_output.csv"
model_path = root_dir / "Week 3" / "demand_forecaster.joblib"

if forecast_path.exists():
    df_forecast = pd.read_csv(forecast_path, parse_dates=["date"])
else:
    df_forecast = pd.DataFrame()


def calculate_safety_stock(
    demand_std: float, lead_time_days: int = 7, service_level: float = 0.95
) -> float:
    z_score = float(norm.ppf(service_level))
    sigma_lead_time = demand_std * np.sqrt(lead_time_days)
    return float(max(0.0, z_score * sigma_lead_time))


def calculate_reorder_point(
    mean_daily_demand: float,
    demand_std: float,
    lead_time_days: int = 7,
    service_level: float = 0.95,
) -> tuple:
    lead_time_demand = mean_daily_demand * lead_time_days
    safety_stock = calculate_safety_stock(
        demand_std, lead_time_days, service_level
    )
    return (
        float(lead_time_demand),
        float(safety_stock),
        float(lead_time_demand + safety_stock),
    )


def calculate_suggested_replenishment(
    current_stock: float,
    reorder_point: float,
    forecast_horizon_demand: float,
    pack_size: int = 6,
    min_order_qty: int = 12,
) -> int:
    net_requirement = reorder_point - current_stock + forecast_horizon_demand
    if net_requirement <= 0:
        return 0
    order_qty = max(net_requirement, float(min_order_qty))
    packs = int(np.ceil(order_qty / pack_size))
    return int(packs * pack_size)


def assess_stockout_risk(
    current_stock: float,
    lead_time_demand: float,
    lead_time_std: float,
    reorder_point: float,
) -> tuple:
    if current_stock < lead_time_demand:
        risk_category = "HIGH"
    elif current_stock < reorder_point:
        risk_category = "MEDIUM"
    else:
        risk_category = "SAFE"

    if lead_time_std <= 1e-4:
        stockout_prob = 1.0 if current_stock < lead_time_demand else 0.0
    else:
        z_val = (current_stock - lead_time_demand) / lead_time_std
        stockout_prob = float(norm.sf(z_val))

    return risk_category, round(float(stockout_prob) * 100, 1)


def evaluate_forecasts(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model"
) -> dict:
    y_pred_clipped = np.clip(y_pred, 0, None)
    mae = float(np.mean(np.abs(y_true - y_pred_clipped)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred_clipped) ** 2)))
    wape = (
        float((np.sum(np.abs(y_true - y_pred_clipped)) / np.sum(y_true)) * 100)
        if np.sum(y_true) > 0
        else 0.0
    )
    bias = (
        float((np.sum(y_pred_clipped - y_true) / np.sum(y_true)) * 100)
        if np.sum(y_true) > 0
        else 0.0
    )
    mask = y_true > 0
    mape = (
        float(
            np.mean(
                np.abs(
                    (y_true[mask] - y_pred_clipped[mask]) / y_true[mask]
                )
            )
            * 100
        )
        if np.any(mask)
        else 0.0
    )
    return {
        "model": model_name,
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "wape_pct": round(wape, 2),
        "mape_nonzero_pct": round(mape, 2),
        "bias_pct": round(bias, 2),
    }


class ForecastRequest(BaseModel):
    item_id: str = Field(
        ...,
        description="Unique product SKU identifier",
        examples=["FOODS_3_090"],
    )
    store_id: str = Field(
        default="CA_1",
        description="Store location identifier",
        examples=["CA_1"],
    )
    horizon_days: int = Field(
        default=28,
        ge=1,
        le=28,
        description="Forecast horizon in days (1 to 28)",
    )


class ReplenishmentRequest(BaseModel):
    item_id: str = Field(
        ...,
        description="Unique product SKU identifier",
        examples=["FOODS_3_090"],
    )
    store_id: str = Field(
        default="CA_1",
        description="Store location identifier",
        examples=["CA_1"],
    )
    current_stock: int = Field(
        ...,
        ge=0,
        description="Current on-hand physical inventory units",
        examples=[45],
    )
    lead_time_days: int = Field(
        default=7,
        ge=1,
        le=30,
        description="Supplier replenishment lead time in days",
    )
    service_level: float = Field(
        default=0.95,
        ge=0.80,
        le=0.999,
        description="Target cycle service level probability",
    )
    pack_size: int = Field(
        default=6,
        ge=1,
        description="Supplier wholesale case pack multiple",
    )
    min_order_qty: int = Field(
        default=12,
        ge=1,
        description="Supplier minimum order quantity",
    )


class ForecastResponse(BaseModel):
    item_id: str
    store_id: str
    horizon_days: int
    forecast_dates: List[str]
    daily_forecasts: List[float]
    total_forecast_demand: float
    mean_daily_forecast: float
    model_used: str = "XGBoost"


class ReplenishmentResponse(BaseModel):
    item_id: str
    store_id: str
    current_stock: int
    lead_time_days: int
    service_level_target: str
    mean_daily_demand: float
    lead_time_demand: float
    safety_stock: float
    reorder_point: float
    stockout_risk: str
    stockout_probability_pct: float
    suggested_order_qty: int
    action_summary: str


@app.get("/health", summary="Health Check")
def health_check():
    return {
        "status": "healthy",
        "service": "demand-forecasting-api",
        "version": "1.0.0",
        "monitored_products_count": (
            int(df_forecast["item_id"].nunique()) if not df_forecast.empty else 0
        ),
    }


@app.get("/products", summary="List Monitored Products and Stores")
def get_products():
    if df_forecast.empty:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Forecast dataset not loaded.",
        )
    items = sorted(df_forecast["item_id"].unique().tolist())
    stores = sorted(df_forecast["store_id"].unique().tolist())
    return {
        "total_items": len(items),
        "items": items,
        "total_stores": len(stores),
        "stores": stores,
    }


@app.get("/metrics", summary="Model Performance Scorecard")
def get_model_metrics():
    if df_forecast.empty:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Forecast dataset not loaded.",
        )
    models_data = [
        ("Naive (Lag 7)", df_forecast["naive_lag7"].values),
        ("Moving Average (7-Day)", df_forecast["moving_avg_7"].values),
        ("Moving Average (28-Day)", df_forecast["moving_avg_28"].values),
        ("Seasonal Naive (Lag 28)", df_forecast["seasonal_naive_28"].values),
        ("Random Forest", df_forecast["pred_random_forest"].values),
        ("LightGBM", df_forecast["pred_lightgbm"].values),
        ("XGBoost (Winner)", df_forecast["best_forecast"].values),
    ]

    scorecard = [
        evaluate_forecasts(df_forecast["sales"].values, p, name)
        for name, p in models_data
    ]
    return {"metrics_scorecard": scorecard}


@app.post(
    "/forecast", response_model=ForecastResponse, summary="Generate Demand Forecast"
)
def generate_forecast(request: ForecastRequest):
    sub = df_forecast[
        (df_forecast["item_id"] == request.item_id)
        & (df_forecast["store_id"] == request.store_id)
    ].sort_values("date")

    if sub.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item '{request.item_id}' at store '{request.store_id}' not found in catalog.",
        )

    horizon_sub = sub.head(request.horizon_days)
    dates = horizon_sub["date"].dt.strftime("%Y-%m-%d").tolist()
    daily_preds = [round(float(v), 2) for v in horizon_sub["best_forecast"].values]
    total_demand = round(float(np.sum(daily_preds)), 2)
    mean_daily = round(float(np.mean(daily_preds)), 2)

    return ForecastResponse(
        item_id=request.item_id,
        store_id=request.store_id,
        horizon_days=request.horizon_days,
        forecast_dates=dates,
        daily_forecasts=daily_preds,
        total_forecast_demand=total_demand,
        mean_daily_forecast=mean_daily,
        model_used="XGBoost",
    )


@app.post(
    "/replenishment",
    response_model=ReplenishmentResponse,
    summary="Generate Replenishment Recommendation",
)
def generate_replenishment(request: ReplenishmentRequest):
    sub = df_forecast[
        (df_forecast["item_id"] == request.item_id)
        & (df_forecast["store_id"] == request.store_id)
    ]

    if sub.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item '{request.item_id}' at store '{request.store_id}' not found.",
        )

    mean_daily = float(sub["best_forecast"].mean())
    std_daily = (
        float(sub["sales"].std())
        if sub["sales"].std() > 0
        else float(sub["best_forecast"].std() + 0.5)
    )

    ltd, ss, rop = calculate_reorder_point(
        mean_daily_demand=mean_daily,
        demand_std=std_daily,
        lead_time_days=request.lead_time_days,
        service_level=request.service_level,
    )

    sigma_ltd = std_daily * np.sqrt(request.lead_time_days)

    risk, prob = assess_stockout_risk(
        current_stock=request.current_stock,
        lead_time_demand=ltd,
        lead_time_std=sigma_ltd,
        reorder_point=rop,
    )

    horizon_demand = float(sub["best_forecast"].sum())
    order_qty = calculate_suggested_replenishment(
        current_stock=request.current_stock,
        reorder_point=rop,
        forecast_horizon_demand=horizon_demand,
        pack_size=request.pack_size,
        min_order_qty=request.min_order_qty,
    )

    if risk == "HIGH":
        summary = f"CRITICAL: Stock ({request.current_stock}) is below lead-time demand ({ltd:.1f}). Place urgent order of {order_qty} units."
    elif risk == "MEDIUM":
        summary = f"REORDER: Stock ({request.current_stock}) is below ROP ({rop:.1f}). Order {order_qty} units to maintain {request.service_level*100:.0f}% service level."
    else:
        summary = f"SAFE: Stock ({request.current_stock}) exceeds ROP ({rop:.1f}). No immediate replenishment required."

    return ReplenishmentResponse(
        item_id=request.item_id,
        store_id=request.store_id,
        current_stock=request.current_stock,
        lead_time_days=request.lead_time_days,
        service_level_target=f"{request.service_level*100:.0f}%",
        mean_daily_demand=round(mean_daily, 2),
        lead_time_demand=round(ltd, 1),
        safety_stock=round(ss, 1),
        reorder_point=round(rop, 1),
        stockout_risk=risk,
        stockout_probability_pct=prob,
        suggested_order_qty=order_qty,
        action_summary=summary,
    )
