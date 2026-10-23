from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm


def calculate_safety_stock(
    demand_std: float, lead_time_days: int = 7, service_level: float = 0.95
) -> float:
    # Calculating normal distribution z-score for target cycle service level.
    z_score = float(norm.ppf(service_level))
    sigma_lead_time = demand_std * np.sqrt(lead_time_days)
    safety_stock = z_score * sigma_lead_time
    return float(max(0.0, safety_stock))


def calculate_reorder_point(
    mean_daily_demand: float,
    demand_std: float,
    lead_time_days: int = 7,
    service_level: float = 0.95,
) -> tuple:
    # Calculating lead time demand and safety stock buffer.
    lead_time_demand = mean_daily_demand * lead_time_days
    safety_stock = calculate_safety_stock(
        demand_std, lead_time_days, service_level
    )
    reorder_point = lead_time_demand + safety_stock
    return float(lead_time_demand), float(safety_stock), float(reorder_point)


def calculate_suggested_replenishment(
    current_stock: float,
    reorder_point: float,
    forecast_horizon_demand: float,
    pack_size: int = 6,
    min_order_qty: int = 12,
) -> int:
    # Evaluating net inventory replenishment requirement.
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
    # Categorizing stockout risk level and computing stockout probability.
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


def generate_replenishment_recommendations(
    forecast_df: pd.DataFrame,
    lead_time_days: int = 7,
    service_level: float = 0.95,
    pack_size: int = 6,
    min_order_qty: int = 12,
) -> pd.DataFrame:
    # Generating item-level replenishment policies.
    item_groups = forecast_df.groupby(["store_id", "item_id"])
    recommendations = []

    for (store_id, item_id), group in item_groups:
        forecast_demand = group["best_forecast"].sum()
        mean_forecast_daily = group["best_forecast"].mean()
        actual_sales_std = (
            group["sales"].std()
            if group["sales"].std() > 0
            else group["best_forecast"].std() + 0.5
        )

        ltd, ss, rop = calculate_reorder_point(
            mean_daily_demand=mean_forecast_daily,
            demand_std=actual_sales_std,
            lead_time_days=lead_time_days,
            service_level=service_level,
        )

        sigma_ltd = actual_sales_std * np.sqrt(lead_time_days)
        simulated_stock = int(
            np.clip(
                round(ltd * np.random.RandomState(hash(item_id) % 10000).uniform(0.5, 1.8)),
                0,
                None,
            )
        )

        risk_level, stockout_prob = assess_stockout_risk(
            current_stock=simulated_stock,
            lead_time_demand=ltd,
            lead_time_std=sigma_ltd,
            reorder_point=rop,
        )

        suggested_order = calculate_suggested_replenishment(
            current_stock=simulated_stock,
            reorder_point=rop,
            forecast_horizon_demand=forecast_demand,
            pack_size=pack_size,
            min_order_qty=min_order_qty,
        )

        recommendations.append(
            {
                "store_id": store_id,
                "item_id": item_id,
                "lead_time_days": lead_time_days,
                "service_level_target": f"{service_level * 100:.0f}%",
                "mean_daily_forecast": round(mean_forecast_daily, 2),
                "horizon_28d_forecast": round(forecast_demand, 1),
                "lead_time_demand": round(ltd, 1),
                "safety_stock": round(ss, 1),
                "reorder_point": round(rop, 1),
                "current_stock": simulated_stock,
                "stockout_risk": risk_level,
                "stockout_probability_pct": stockout_prob,
                "suggested_order_qty": suggested_order,
            }
        )

    df_rec = pd.DataFrame(recommendations)
    return df_rec.sort_values(
        by=["stockout_probability_pct", "suggested_order_qty"], ascending=False
    ).reset_index(drop=True)


def run_inventory_optimization(
    forecast_path: str = "Week 3/forecast_output.csv",
    output_path: str = "Week 3/replenishment_recommendations.csv",
    lead_time_days: int = 7,
    service_level: float = 0.95,
) -> pd.DataFrame:
    # Locating forecast outputs dataset.
    fc_file = Path(forecast_path)
    if not fc_file.exists():
        fc_file = Path("../Week 3/forecast_output.csv")

    # Loading forecast outputs generated by machine learning model.
    print(f"Loading forecast dataset from {fc_file}...")
    df_forecast = pd.read_csv(fc_file, parse_dates=["date"])

    # Generating inventory policies and purchase order quantities.
    print(
        f"Computing safety stock, ROP, and replenishment orders (Lead time: {lead_time_days}d, Service level: {service_level*100}%)..."
    )
    df_recs = generate_replenishment_recommendations(
        df_forecast,
        lead_time_days=lead_time_days,
        service_level=service_level,
        pack_size=6,
        min_order_qty=12,
    )

    # Saving replenishment recommendations to output path.
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_recs.to_csv(output_path, index=False)
    print(f"Replenishment recommendations written to {output_path}")

    # Displaying sample recommendations table.
    print("\nTop 5 Stockout Risk Items Requiring Action:")
    print(
        df_recs[
            [
                "item_id",
                "current_stock",
                "lead_time_demand",
                "safety_stock",
                "reorder_point",
                "stockout_risk",
                "suggested_order_qty",
            ]
        ].head(5).to_string(index=False)
    )

    return df_recs


if __name__ == "__main__":
    run_inventory_optimization()
