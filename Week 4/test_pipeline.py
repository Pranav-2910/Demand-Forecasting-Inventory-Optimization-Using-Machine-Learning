from pathlib import Path
import sys
import importlib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Adding project root to path for imports.
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

# Importing dynamically from Week 4 module.
api_mod = importlib.import_module("Week 4.api")
app = api_mod.app
assess_stockout_risk = api_mod.assess_stockout_risk
calculate_reorder_point = api_mod.calculate_reorder_point
calculate_safety_stock = api_mod.calculate_safety_stock
calculate_suggested_replenishment = api_mod.calculate_suggested_replenishment
evaluate_forecasts = api_mod.evaluate_forecasts


def test_metrics_calculation():
    y_true = np.array([10.0, 20.0, 0.0, 30.0])
    y_pred = np.array([12.0, 18.0, 2.0, 30.0])

    res = evaluate_forecasts(y_true, y_pred, model_name="TestModel")
    assert res["mae"] == 1.5, f"Expected MAE 1.5, got {res['mae']}"
    assert np.isclose(res["rmse"], np.sqrt(3.0), atol=0.01)
    assert np.isclose(res["wape_pct"], 10.0, atol=0.01)
    assert np.isclose(res["bias_pct"], (2.0 / 60.0) * 100, atol=0.01)


def test_safety_stock_and_reorder_point():
    mean_d = 10.0
    std_d = 2.0
    lead_time = 4
    service_level = 0.95

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
    rop = 50.0
    horizon_demand = 100.0
    pack_size = 6
    min_order = 12

    # Case 1: Low stock triggers order.
    curr_stock = 30.0
    order_qty = calculate_suggested_replenishment(
        current_stock=curr_stock,
        reorder_point=rop,
        forecast_horizon_demand=horizon_demand,
        pack_size=pack_size,
        min_order_qty=min_order,
    )
    assert order_qty == 120, f"Expected 120, got {order_qty}"

    # Case 2: Ample stock requires zero order.
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
    ltd = 50.0
    lead_time_std = 10.0
    rop = 66.45

    risk_high, prob_high = assess_stockout_risk(
        current_stock=30.0,
        lead_time_demand=ltd,
        lead_time_std=lead_time_std,
        reorder_point=rop,
    )
    assert risk_high == "HIGH"
    assert prob_high > 90.0

    risk_safe, prob_safe = assess_stockout_risk(
        current_stock=80.0,
        lead_time_demand=ltd,
        lead_time_std=lead_time_std,
        reorder_point=rop,
    )
    assert risk_safe == "SAFE"
    assert prob_safe < 5.0


def test_api_endpoints():
    client = TestClient(app)

    # Health check.
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "healthy"

    # Products catalog.
    prod_resp = client.get("/products")
    assert prod_resp.status_code == 200
    products = prod_resp.json()["items"]
    assert len(products) > 0
    sample_item = products[0]

    # Metrics scorecard.
    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    assert len(metrics_resp.json()["metrics_scorecard"]) > 0

    # Forecast generation.
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

    # Replenishment endpoint.
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
