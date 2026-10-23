from pathlib import Path
import numpy as np
import pandas as pd


def load_raw_calendar(filepath: str = "data/raw/calendar.csv") -> pd.DataFrame:
    # Reading the calendar lookup table from the raw data directory.
    calendar_path = Path(filepath)
    df_calendar = pd.read_csv(calendar_path)

    # Converting date column to standard datetime format.
    df_calendar["date"] = pd.to_datetime(df_calendar["date"])
    return df_calendar


def load_raw_prices(
    filepath: str = "data/raw/sell_prices.csv",
    store_ids: list = None,
    item_ids: list = None,
) -> pd.DataFrame:
    # Reading the weekly item selling prices table.
    prices_path = Path(filepath)
    df_prices = pd.read_csv(prices_path)

    # Filtering prices for specific stores if provided.
    if store_ids is not None:
        df_prices = df_prices[df_prices["store_id"].isin(store_ids)]

    # Filtering prices for specific items if provided.
    if item_ids is not None:
        df_prices = df_prices[df_prices["item_id"].isin(item_ids)]

    return df_prices


def load_raw_sales(
    filepath: str = "data/raw/sales_train_validation.csv",
    store_ids: list = None,
    cat_ids: list = None,
    top_n_items: int = None,
) -> pd.DataFrame:
    # Reading product sales history from the raw dataset.
    sales_path = Path(filepath)
    df_sales = pd.read_csv(sales_path)

    # Filtering sales by store identifier if requested.
    if store_ids is not None:
        df_sales = df_sales[df_sales["store_id"].isin(store_ids)]

    # Filtering sales by category identifier if requested.
    if cat_ids is not None:
        df_sales = df_sales[df_sales["cat_id"].isin(cat_ids)]

    # Selecting top selling items based on total sales volume when requested.
    if top_n_items is not None and top_n_items < len(df_sales):
        day_cols = [c for c in df_sales.columns if c.startswith("d_")]
        sales_sums = df_sales[day_cols].sum(axis=1)
        top_indices = sales_sums.nlargest(top_n_items).index
        df_sales = df_sales.loc[top_indices].copy()

    return df_sales


def transform_sales_to_long(
    df_sales: pd.DataFrame,
    df_calendar: pd.DataFrame,
    df_prices: pd.DataFrame = None,
) -> pd.DataFrame:
    # Identifying identifier columns and daily sales count columns.
    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [col for col in df_sales.columns if col.startswith("d_")]

    # Melting sales dataframe from wide format into longitudinal format.
    df_long = pd.melt(
        df_sales,
        id_vars=id_vars,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )

    # Merging calendar attributes to enrich records with calendar dates and events.
    calendar_cols = [
        "date",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "d",
        "event_name_1",
        "event_type_1",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    df_merged = pd.merge(df_long, df_calendar[calendar_cols], on="d", how="left")

    # Merging weekly sell prices when price dataset is supplied.
    if df_prices is not None:
        df_merged = pd.merge(
            df_merged,
            df_prices,
            on=["store_id", "item_id", "wm_yr_wk"],
            how="left",
        )

    # Sorting records chronologically by item, store, and date.
    df_merged = df_merged.sort_values(
        by=["store_id", "item_id", "date"]
    ).reset_index(drop=True)

    # Flagging active product selling windows to separate true zeros from pre-launch periods.
    df_merged = flag_product_activity(df_merged)

    return df_merged


def flag_product_activity(df: pd.DataFrame) -> pd.DataFrame:
    # Determining the first selling day for each item to identify pre-launch records.
    # Grouping by item to locate first non-zero sales date.
    first_sales = (
        df[df["sales"] > 0]
        .groupby(["store_id", "item_id"])["date"]
        .min()
        .reset_index()
        .rename(columns={"date": "first_sale_date"})
    )

    # Merging first sale date back to main dataframe.
    df = pd.merge(df, first_sales, on=["store_id", "item_id"], how="left")

    # Marking records prior to first sale as inactive pre-launch status.
    df["is_active"] = df["date"] >= df["first_sale_date"]

    # Flagging potential stockout periods where active items with historical sales register 0 sales.
    # Detecting zero sales during active periods when price is present.
    has_price = df["sell_price"].notna() if "sell_price" in df.columns else True
    df["is_zero_sales"] = df["sales"] == 0
    df["potential_stockout"] = df["is_active"] & df["is_zero_sales"] & has_price

    return df


def generate_interim_dataset(
    raw_sales_path: str = "data/raw/sales_train_validation.csv",
    raw_calendar_path: str = "data/raw/calendar.csv",
    raw_prices_path: str = "data/raw/sell_prices.csv",
    output_path: str = "data/interim/sales_interim.csv",
    store_ids: list = ["CA_1"],
    top_n_items: int = 50,
) -> pd.DataFrame:
    # Loading calendar dataset.
    df_calendar = load_raw_calendar(raw_calendar_path)

    # Loading filtered sales dataset for representative stores and items.
    df_sales = load_raw_sales(
        filepath=raw_sales_path,
        store_ids=store_ids,
        top_n_items=top_n_items,
    )

    # Extracting unique item ids to load corresponding prices.
    selected_items = df_sales["item_id"].unique().tolist()
    df_prices = load_raw_prices(
        filepath=raw_prices_path,
        store_ids=store_ids,
        item_ids=selected_items,
    )

    # Transforming wide sales dataset into structured longitudinal format.
    df_interim = transform_sales_to_long(df_sales, df_calendar, df_prices)

    # Saving cleaned interim dataset to disk for downstream feature engineering.
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_interim.to_csv(out_file, index=False)

    return df_interim
