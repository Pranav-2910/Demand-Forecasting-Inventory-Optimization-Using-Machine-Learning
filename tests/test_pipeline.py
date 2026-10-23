from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Adding project root to path for imports.
sys.path.append(str(Path(".").resolve()))

from app.api import app
from src.baselines import generate_baseline_predictions
from src.inventory import (
    assess_stockout_risk,
    calculate_reorder_point,
    calculate_safety_stock,
    calculate_suggested_replenishment,
)
from src.metrics import (
    calculate_forecast_bias,
    calculate_mae,
    calculate_rmse,
    calculate_wape,
)
from src.splitting import get_split_cutoff_dates, split_chronologically


def test_metrics_calculation():
    # Testing metric calculation accuracy on deterministic vectors.
    y_true = np.array([10.0, 20.0, 0.0, 30.0])
    y_pred = np.array([12.0, 18.0, 2.0, 30.0])

    mae = calculate_mae(y_true, y_pred)
    assert np.isclose(mae, 1.5), f"Expected MAE 1.5, got {mae}"

    rmse = calculate_rmse(y_true, y_pred)
    assert np.isclose(rmse, np.sqrt(3.0)), f"Unexpected RMSE: {rmse}"

    wape = calculate_wape(y_true, y_pred)
    assert np.isclose(wape, (6.0 / 60.0) * 100), f"Expected WAPE 10.0, got {wape}"

    bias = calculate_forecast_bias(y_true, y_pred)
    assert np.isclose(bias, (2.0 / 60.0) * 100), f"Unexpected Bias: {bias}"


def test_safety_stock_and_reorder_point():
    # Testing inventory safety stock and reorder point calculations.
    mean_d = 10.0
    std_d = 2.0
    lead_time = 4  # sqrt(4) = 2
    service_level = 0.95  # z ~ 1.64485

    ltd, ss, rop = calculate_reorder_point(
        mean_daily_demand=mean_d,
        demand_std=std_d,
        lead_time_days=lead_time,
        service_level=service_level,
    )

    assert ltd == 40.0, f"Expected LTD 40.0, got {ltd}"
    expected_ss = 1.6448536269514722 * (2.0 * 2.0)
    assert np.isclose(ss, expected_ss, atol=0.01), f"Unexpected SS: {ss}"
    assert np.isclose(rop, 40.0 + expected_ss, atol=0.01), f"Unexpected ROP: {rop}"


def test_suggested_replenishment_orders():
    # Testing replenishment ordering logic and pack size rounding.
    rop = 50.0
    horizon_demand = 100.0
    pack_size = 6
    min_order = 12

    # Case 1: Low stock below ROP triggers replenishment order.
    curr_stock = 30.0
    order_qty = calculate_suggested_replenishment(
        current_stock=curr_stock,
        reorder_point=rop,
        forecast_horizon_demand=horizon_demand,
        pack_size=pack_size,
        min_order_qty=min_order,
    )
    # Net requirement: 50 - 30 + 100 = 120 (already multiple of 6).
    assert order_qty == 120, f"Expected 120, got {order_qty}"

    # Case 2: Excessive stock requires zero replenishment order.
    ample_stock = 200.0
    zero_order = calculate_suggested_replenishment(
        current_stock=ample_stock,
        reorder_point=rop,
        forecast_horizon_demand=horizon_demand,
        pack_size=pack_size,
        min_order_qty=min_order,
    )
    assert zero_order == 0, f"Expected 0, got {zero_order}"


def test_stockout_risk_assessment():
    # Testing categorical and probabilistic stockout risk classifications.
    ltd = 50.0
    lead_time_std = 10.0
    rop = 66.45

    # High risk when current stock is below lead time demand.
    risk_high, prob_high = assess_stockout_risk(
        current_stock=30.0,
        lead_time_demand=ltd,
        lead_time_std=lead_time_std,
        reorder_point=rop,
    )
    assert risk_high == "HIGH"
    assert prob_high > 90.0

    # Safe status when current stock exceeds reorder point.
    risk_safe, prob_safe = assess_stockout_risk(
        current_stock=80.0,
        lead_time_demand=ltd,
        lead_time_std=lead_time_std,
        reorder_point=rop,
    )
    assert risk_safe == "SAFE"
    assert prob_safe < 5.0


def test_chronological_splitting():
    # Testing chronological date partitioning guarantees.
    dates = pd.date_range("2015-01-01", periods=100, freq="D")
    df_dummy = pd.DataFrame({"date": dates, "sales": np.ones(100)})

    df_train, df_val, df_test, cutoffs = split_chronologically(
        df_dummy, val_days=14, test_days=14
    )

    assert len(df_test) == 14
    assert len(df_val) == 14
    assert len(df_train) == 100 - 28
    assert df_train["date"].max() < df_val["date"].min()
    assert df_val["date"].max() < df_test["date"].min()


def test_api_endpoints():
    # Testing FastAPI endpoints using TestClient.
    client = TestClient(app)

    # Testing health endpoint.
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "healthy"

    # Testing products catalog endpoint.
    prod_resp = client.get("/products")
    assert prod_resp.status_code == 200
    products = prod_resp.json()["items"]
    assert len(products) > 0
    sample_item = products[0]

    # Testing metrics scorecard endpoint.
    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    assert len(metrics_resp.json()["metrics_scorecard"]) > 0

    # Testing forecast generation endpoint.
    fc_payload = {
        "item_id": sample_item,
        "store_id": "CA_1",
        "horizon_days": 14,
    }
    fc_resp = client.post("/forecast", json=fc_payload)
    assert fc_resp.status_code == 200
    fc_data = fc_resp.json()
    assert fc_data["item_id"] == sample_item
    assert len(fc_data["daily_forecasts"]) == 14
    assert fc_data["total_forecast_demand"] >= 0

    # Testing inventory replenishment endpoint.
    rep_payload = {
        "item_id": sample_item,
        "store_id": "CA_1",
        "current_stock": 25,
        "lead_time_days": 7,
        "service_level": 0.95,
        "pack_size": 6,
    }
    rep_resp = client.post("/replenishment", json=rep_payload)
    assert rep_resp.status_code == 200
    rep_data = rep_resp.json()
    assert rep_data["stockout_risk"] in ["HIGH", "MEDIUM", "SAFE"]
    assert rep_data["suggested_order_qty"] % 6 == 0
