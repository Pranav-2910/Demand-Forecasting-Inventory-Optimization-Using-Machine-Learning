from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configuring Streamlit dashboard page layout and styling.
st.set_page_config(
    page_title="Demand Forecasting & Inventory Optimization",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Locating project root directory dynamically.
root_dir = Path(".").resolve()
if not (root_dir / "data").exists() and (root_dir / ".." / "data").exists():
    root_dir = (root_dir / "..").resolve()

forecast_path = root_dir / "data" / "processed" / "forecast_output.csv"
replenishment_path = (
    root_dir / "data" / "processed" / "replenishment_recommendations.csv"
)
features_path = root_dir / "data" / "processed" / "features_engineered.csv"


@st.cache_data
def load_dashboard_data():
    # Ingesting processed datasets for dashboard visualization.
    df_fc = pd.read_csv(forecast_path, parse_dates=["date"])
    df_rp = pd.read_csv(replenishment_path)
    df_feat = pd.read_csv(features_path, parse_dates=["date"])
    return df_fc, df_rp, df_feat


# Loading cached analytical datasets.
df_forecast, df_replenishment, df_features = load_dashboard_data()

# Rendering navigation sidebar.
st.sidebar.title("📦 Supply Chain Analytics")
st.sidebar.markdown("**Demand Forecasting & Inventory Optimization System**")
page = st.sidebar.radio(
    "Navigation Menu",
    [
        "Demand Overview",
        "Product / Store Forecasts",
        "Inventory Risk & Replenishment",
        "Model Performance & Benchmarks",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Retail Operations Analytics • Walmart M5 Case Study")

# ==============================================================================
# PAGE 1: DEMAND OVERVIEW
# ==============================================================================
if page == "Demand Overview":
    st.title("📊 Enterprise Demand Overview")
    st.markdown(
        "Real-time visibility into out-of-sample actual sales, winning model forecasts, and aggregate forecast errors."
    )

    # Filtering data by category.
    categories = ["All"] + sorted(df_forecast["cat_id"].unique().tolist())
    selected_cat = st.selectbox("Filter by Category", categories)

    filtered_df = df_forecast.copy()
    if selected_cat != "All":
        filtered_df = filtered_df[filtered_df["cat_id"] == selected_cat]

    # Computing aggregate operational metrics.
    total_actual = filtered_df["sales"].sum()
    total_forecast = filtered_df["best_forecast"].sum()
    total_abs_error = filtered_df["abs_error"].sum()
    overall_wape = (
        (total_abs_error / total_actual * 100) if total_actual > 0 else 0
    )
    overall_bias = (
        ((total_forecast - total_actual) / total_actual * 100)
        if total_actual > 0
        else 0
    )

    # Displaying high-level KPI cards.
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Actual Demand (Units)", f"{total_actual:,.0f}")
    kpi2.metric(
        "Forecast Demand (Units)",
        f"{total_forecast:,.0f}",
        delta=f"{overall_bias:+.1f}% Bias",
    )
    kpi3.metric("Aggregate WAPE", f"{overall_wape:.1f}%")
    kpi4.metric(
        "Forecast Bias",
        f"{overall_bias:+.2f}%",
        delta_color="inverse" if abs(overall_bias) > 5 else "normal",
    )
    kpi5.metric("Products Monitored", f"{filtered_df['item_id'].nunique()}")

    st.markdown("---")

    # Aggregating daily actuals vs forecast.
    daily_trend = (
        filtered_df.groupby("date")
        .agg(
            actual_sales=("sales", "sum"),
            forecast_sales=("best_forecast", "sum"),
            error=("forecast_error", "sum"),
        )
        .reset_index()
    )

    # Rendering Plotly time series chart for demand overview.
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily_trend["date"],
            y=daily_trend["actual_sales"],
            mode="lines+markers",
            name="Actual Demand",
            line=dict(color="#1f77b4", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily_trend["date"],
            y=daily_trend["forecast_sales"],
            mode="lines+markers",
            name="XGBoost Forecast",
            line=dict(color="#d62728", width=3, dash="dot"),
        )
    )
    fig.update_layout(
        title="Aggregate Daily Actual Demand vs ML Forecast (Test Period)",
        xaxis_title="Date",
        yaxis_title="Units Sold",
        hovermode="x unified",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Displaying daily error distribution.
    col_left, col_right = st.columns(2)
    with col_left:
        fig_err = px.bar(
            daily_trend,
            x="date",
            y="error",
            title="Daily Net Forecast Error (Forecast - Actual)",
            color="error",
            color_continuous_scale="RdBu_r",
        )
        st.plotly_chart(fig_err, use_container_width=True)

    with col_right:
        # Displaying cumulative sales comparison.
        daily_trend["cum_actual"] = daily_trend["actual_sales"].cumsum()
        daily_trend["cum_forecast"] = daily_trend["forecast_sales"].cumsum()
        fig_cum = go.Figure()
        fig_cum.add_trace(
            go.Scatter(
                x=daily_trend["date"],
                y=daily_trend["cum_actual"],
                name="Cumulative Actuals",
                line=dict(color="#2ca02c", width=2.5),
            )
        )
        fig_cum.add_trace(
            go.Scatter(
                x=daily_trend["date"],
                y=daily_trend["cum_forecast"],
                name="Cumulative Forecast",
                line=dict(color="#ff7f0e", width=2.5, dash="dash"),
            )
        )
        fig_cum.update_layout(
            title="Cumulative Demand Tracking",
            xaxis_title="Date",
            yaxis_title="Cumulative Units",
            height=380,
        )
        st.plotly_chart(fig_cum, use_container_width=True)

# ==============================================================================
# PAGE 2: PRODUCT / STORE FORECASTS
# ==============================================================================
elif page == "Product / Store Forecasts":
    st.title("🔍 Granular Product & Store Forecast Trajectories")
    st.markdown(
        "Inspect individual SKU demand dynamics, compare ML predictions against heuristic baselines, and review price elasticity."
    )

    # Selecting item to inspect.
    available_items = sorted(df_forecast["item_id"].unique().tolist())
    selected_item = st.selectbox(
        "Select Product SKU", available_items, index=0
    )

    item_df = df_forecast[df_forecast["item_id"] == selected_item].sort_values(
        "date"
    )
    item_hist = df_features[
        df_features["item_id"] == selected_item
    ].sort_values("date")

    # Displaying product metadata.
    cat = item_df["cat_id"].iloc[0]
    dept = item_df["dept_id"].iloc[0]
    store = item_df["store_id"].iloc[0]
    price = item_df["sell_price"].mean()
    item_actual_sum = item_df["sales"].sum()
    item_fc_sum = item_df["best_forecast"].sum()
    item_wape = (
        (item_df["abs_error"].sum() / item_actual_sum * 100)
        if item_actual_sum > 0
        else 0
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Category / Dept", f"{cat} • {dept}")
    c2.metric("Store Location", store)
    c3.metric("Mean Sell Price", f"${price:.2f}")
    c4.metric(
        "Total 28d Demand",
        f"{item_actual_sum} Units",
        delta=f"Forecast: {item_fc_sum:.0f}",
    )
    c5.metric("Product WAPE", f"{item_wape:.1f}%")

    st.markdown("---")

    # Plotting granular forecast trajectory vs baselines.
    fig_traj = go.Figure()
    fig_traj.add_trace(
        go.Scatter(
            x=item_df["date"],
            y=item_df["sales"],
            mode="lines+markers",
            name="Ground Truth (Actuals)",
            line=dict(color="#000000", width=2.5),
            marker=dict(size=6),
        )
    )
    fig_traj.add_trace(
        go.Scatter(
            x=item_df["date"],
            y=item_df["best_forecast"],
            mode="lines+markers",
            name="XGBoost Forecast (Winner)",
            line=dict(color="#d62728", width=2.5),
            marker=dict(symbol="square", size=6),
        )
    )
    fig_traj.add_trace(
        go.Scatter(
            x=item_df["date"],
            y=item_df["moving_avg_7"],
            mode="lines",
            name="Moving Average (7d)",
            line=dict(color="#2ca02c", dash="dash"),
        )
    )
    fig_traj.add_trace(
        go.Scatter(
            x=item_df["date"],
            y=item_df["naive_lag7"],
            mode="lines",
            name="Naive (Lag 7)",
            line=dict(color="#1f77b4", dash="dot"),
        )
    )
    fig_traj.update_layout(
        title=f"Detailed 28-Day Demand Trajectory — SKU: {selected_item}",
        xaxis_title="Date",
        yaxis_title="Units Sold",
        hovermode="x unified",
        height=450,
    )
    st.plotly_chart(fig_traj, use_container_width=True)

    # Plotting historical seasonality profile.
    st.subheader("Historical Demand Profiling (Last 120 Days)")
    hist_recent = item_hist.tail(120)
    fig_hist = px.line(
        hist_recent,
        x="date",
        y="sales",
        title="Historical Daily Sales Velocity",
        labels={"sales": "Units Sold", "date": "Date"},
    )
    fig_hist.update_traces(line_color="#4b0082")
    st.plotly_chart(fig_hist, use_container_width=True)

# ==============================================================================
# PAGE 3: INVENTORY RISK & REPLENISHMENT
# ==============================================================================
elif page == "Inventory Risk & Replenishment":
    st.title("🛡️ Inventory Optimization & Replenishment Decision Support")
    st.markdown(
        "Convert probabilistic machine learning forecasts into optimal safety stocks, reorder points (ROP), and automated replenishment orders."
    )

    # Interactive policy sliders.
    st.sidebar.subheader("Policy Configuration Parameters")
    lead_time = st.sidebar.slider(
        "Supplier Lead Time (Days)",
        min_value=3,
        max_value=21,
        value=7,
        step=1,
    )
    service_level = st.sidebar.slider(
        "Target Cycle Service Level (%)",
        min_value=85,
        max_value=99,
        value=95,
        step=1,
    )
    pack_size = st.sidebar.selectbox("Wholesale Pack Size", [6, 12, 24], index=0)

    # Recalculating dynamic inventory policies with user parameters.
    from src.inventory import generate_replenishment_recommendations

    df_dynamic_recs = generate_replenishment_recommendations(
        df_forecast,
        lead_time_days=lead_time,
        service_level=service_level / 100.0,
        pack_size=pack_size,
        min_order_qty=pack_size * 2,
    )

    # Risk tier metrics.
    high_risk_count = (df_dynamic_recs["stockout_risk"] == "HIGH").sum()
    med_risk_count = (df_dynamic_recs["stockout_risk"] == "MEDIUM").sum()
    safe_count = (df_dynamic_recs["stockout_risk"] == "SAFE").sum()
    total_order_units = df_dynamic_recs["suggested_order_qty"].sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🚨 High Stockout Risk SKUs", f"{high_risk_count}")
    m2.metric("⚠️ Reorder Window SKUs", f"{med_risk_count}")
    m3.metric("✅ Sufficient Buffer SKUs", f"{safe_count}")
    m4.metric("Suggested Total Order", f"{total_order_units:,} Units")

    st.markdown("---")

    # Plotting stock vs ROP for highest risk SKUs.
    st.subheader("Current Stock vs Reorder Point (ROP) for Top Priority SKUs")
    top_risk = df_dynamic_recs.head(15)

    fig_rop = go.Figure()
    fig_rop.add_trace(
        go.Bar(
            x=top_risk["item_id"],
            y=top_risk["current_stock"],
            name="Current On-Hand Stock",
            marker_color="#3498db",
        )
    )
    fig_rop.add_trace(
        go.Bar(
            x=top_risk["item_id"],
            y=top_risk["reorder_point"],
            name="Reorder Point (ROP)",
            marker_color="#e74c3c",
        )
    )
    fig_rop.update_layout(
        barmode="group",
        xaxis_title="Product SKU",
        yaxis_title="Units",
        height=420,
    )
    st.plotly_chart(fig_rop, use_container_width=True)

    # Filterable replenishment recommendation table.
    st.subheader("Actionable Replenishment Recommendations Table")
    risk_filter = st.multiselect(
        "Filter by Risk Level",
        ["HIGH", "MEDIUM", "SAFE"],
        default=["HIGH", "MEDIUM"],
    )
    table_view = df_dynamic_recs[
        df_dynamic_recs["stockout_risk"].isin(risk_filter)
    ]

    st.dataframe(
        table_view[
            [
                "item_id",
                "store_id",
                "current_stock",
                "lead_time_demand",
                "safety_stock",
                "reorder_point",
                "stockout_risk",
                "stockout_probability_pct",
                "suggested_order_qty",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

# ==============================================================================
# PAGE 4: MODEL PERFORMANCE & BENCHMARKS
# ==============================================================================
elif page == "Model Performance & Benchmarks":
    st.title("📈 Model Evaluation & Baseline Comparison Scorecard")
    st.markdown(
        "Comprehensive out-of-sample benchmark of machine learning models against heuristic baselines using chronological evaluation."
    )

    # Computing metrics table across models.
    from src.metrics import evaluate_forecasts

    models_data = [
        ("Naive (Lag 7)", df_forecast["naive_lag7"].values),
        ("Moving Average (7-Day)", df_forecast["moving_avg_7"].values),
        ("Moving Average (28-Day)", df_forecast["moving_avg_28"].values),
        ("Seasonal Naive (Lag 28)", df_forecast["seasonal_naive_28"].values),
        ("Random Forest", df_forecast["pred_random_forest"].values),
        ("LightGBM", df_forecast["pred_lightgbm"].values),
        ("XGBoost (Winner)", df_forecast["pred_xgboost"].values),
    ]

    scorecard_list = [
        evaluate_forecasts(df_forecast["sales"].values, p, name)
        for name, p in models_data
    ]
    df_perf = pd.DataFrame(scorecard_list)

    st.dataframe(
        df_perf.rename(
            columns={
                "model": "Model Name",
                "mae": "MAE (Units)",
                "rmse": "RMSE",
                "wape_pct": "WAPE (%)",
                "mape_nonzero_pct": "Non-Zero MAPE (%)",
                "bias_pct": "Forecast Bias (%)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        # WAPE comparison chart.
        fig_wape = px.bar(
            df_perf,
            x="model",
            y="wape_pct",
            color="wape_pct",
            color_continuous_scale="Viridis_r",
            title="WAPE (%) Comparison — Lower is Better",
            labels={"wape_pct": "WAPE (%)", "model": "Model"},
        )
        st.plotly_chart(fig_wape, use_container_width=True)

    with col2:
        # Residual error histogram.
        fig_res = px.histogram(
            df_forecast,
            x="forecast_error",
            nbins=40,
            title="Residual Error Distribution (XGBoost)",
            labels={"forecast_error": "Forecast Error (Units)"},
            color_discrete_sequence=["#2b5c8f"],
        )
        st.plotly_chart(fig_res, use_container_width=True)
