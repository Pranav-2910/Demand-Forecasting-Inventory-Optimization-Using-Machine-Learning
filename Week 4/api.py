from pathlib import Path
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.inventory import (
    assess_stockout_risk,
    calculate_reorder_point,
    calculate_suggested_replenishment,
)
from src.metrics import evaluate_forecasts

# Initializing FastAPI web application instance.
app = FastAPI(
    title="Demand Forecasting & Inventory Replenishment API",
    description="Enterprise REST API serving production machine learning demand forecasts and automated inventory replenishment recommendations.",
    version="1.0.0",
)

# Locating project root directory dynamically.
root_dir = Path(".").resolve()
if not (root_dir / "data").exists() and (root_dir / ".." / "data").exists():
    root_dir = (root_dir / "..").resolve()

forecast_path = root_dir / "data" / "processed" / "forecast_output.csv"
model_path = root_dir / "models" / "demand_forecaster.joblib"

# Loading forecast dataset into memory.
if forecast_path.exists():
    df_forecast = pd.read_csv(forecast_path, parse_dates=["date"])
else:
    df_forecast = pd.DataFrame()


# Defining Pydantic request models.
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


# Defining Pydantic response models.
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
    # Checking API operational status and loaded dataset availability.
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
    # Returning catalog of monitored products and stores.
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
    # Computing and returning model evaluation scorecard.
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
    # Validating item and store presence in forecast registry.
    sub = df_forecast[
        (df_forecast["item_id"] == request.item_id)
        & (df_forecast["store_id"] == request.store_id)
    ].sort_values("date")

    if sub.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item '{request.item_id}' at store '{request.store_id}' not found in catalog.",
        )

    # Slicing predictions for requested horizon.
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
    # Validating item and store presence in dataset.
    sub = df_forecast[
        (df_forecast["item_id"] == request.item_id)
        & (df_forecast["store_id"] == request.store_id)
    ]

    if sub.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item '{request.item_id}' at store '{request.store_id}' not found.",
        )

    # Calculating statistical demand parameters.
    mean_daily = float(sub["best_forecast"].mean())
    std_daily = (
        float(sub["sales"].std())
        if sub["sales"].std() > 0
        else float(sub["best_forecast"].std() + 0.5)
    )

    # Calculating safety stock and reorder point.
    ltd, ss, rop = calculate_reorder_point(
        mean_daily_demand=mean_daily,
        demand_std=std_daily,
        lead_time_days=request.lead_time_days,
        service_level=request.service_level,
    )

    sigma_ltd = std_daily * np.sqrt(request.lead_time_days)

    # Assessing stockout risk status.
    risk, prob = assess_stockout_risk(
        current_stock=request.current_stock,
        lead_time_demand=ltd,
        lead_time_std=sigma_ltd,
        reorder_point=rop,
    )

    # Calculating suggested replenishment order quantity.
    horizon_demand = float(sub["best_forecast"].sum())
    order_qty = calculate_suggested_replenishment(
        current_stock=request.current_stock,
        reorder_point=rop,
        forecast_horizon_demand=horizon_demand,
        pack_size=request.pack_size,
        min_order_qty=request.min_order_qty,
    )

    # Generating human-readable executive summary.
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
