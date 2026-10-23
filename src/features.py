from pathlib import Path
import numpy as np
import pandas as pd


def extract_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    # Creating date-based calendar attributes from datetime series.
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # Creating indicator flag for weekend days (Saturday and Sunday).
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Creating binary flag indicating presence of holiday or promotional event.
    df["has_event"] = df["event_name_1"].notna().astype(int)

    # Encoding event categories into simple integer representations.
    event_type_mapping = {
        "Sporting": 1,
        "Cultural": 2,
        "National": 3,
        "Religious": 4,
    }
    df["event_type_code"] = (
        df["event_type_1"].map(event_type_mapping).fillna(0).astype(int)
    )

    return df


def extract_price_features(df: pd.DataFrame) -> pd.DataFrame:
    # Calculating historical item pricing baselines without future target leakage.
    price_stats = (
        df.groupby(["store_id", "item_id"])["sell_price"]
        .agg(["mean", "max", "min"])
        .reset_index()
        .rename(
            columns={
                "mean": "item_mean_price",
                "max": "item_max_price",
                "min": "item_min_price",
            }
        )
    )

    # Merging historical price baselines back into main dataframe.
    df = pd.merge(df, price_stats, on=["store_id", "item_id"], how="left")

    # Imputing missing sell prices with historical item mean price.
    df["sell_price"] = df["sell_price"].fillna(df["item_mean_price"])

    # Computing relative price difference compared to item mean.
    df["price_diff_to_mean"] = df["sell_price"] - df["item_mean_price"]

    # Calculating discount ratio relative to item historical maximum price.
    df["discount_ratio"] = np.where(
        df["item_max_price"] > 0,
        df["sell_price"] / df["item_max_price"],
        1.0,
    )

    # Flagging promotional discounts where selling price is below 95% of regular max price.
    df["is_discounted"] = (df["discount_ratio"] < 0.95).astype(int)

    return df


def extract_lag_and_rolling_features(
    df: pd.DataFrame,
    lag_days: list = [7, 14, 21, 28, 35, 42],
    rolling_windows: list = [7, 14, 28],
) -> pd.DataFrame:
    # Ensuring dataframe is sorted strictly in chronological order by product and store.
    df = df.sort_values(by=["store_id", "item_id", "date"]).reset_index(
        drop=True
    )

    # Generating chronological lag features strictly using past observations.
    for lag in lag_days:
        df[f"sales_lag_{lag}"] = df.groupby(["store_id", "item_id"])[
            "sales"
        ].shift(lag)

    # Generating non-leaking rolling window summary statistics anchored at lag 7.
    # Anchoring rolling computations at lag 7 ensures features are available at forecast time.
    for window in rolling_windows:
        grouped_lag7 = df.groupby(["store_id", "item_id"])["sales_lag_7"]
        df[f"rolling_mean_{window}"] = grouped_lag7.transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        )
        df[f"rolling_std_{window}"] = grouped_lag7.transform(
            lambda s: s.rolling(window=window, min_periods=1).std()
        ).fillna(0)
        df[f"rolling_max_{window}"] = grouped_lag7.transform(
            lambda s: s.rolling(window=window, min_periods=1).max()
        )

    return df


def build_feature_pipeline(
    df_raw: pd.DataFrame, drop_nas: bool = True
) -> pd.DataFrame:
    # Applying calendar feature extraction.
    df = extract_calendar_features(df_raw.copy())

    # Applying pricing and promotional feature extraction.
    df = extract_price_features(df)

    # Applying non-leaking lag and rolling statistics extraction.
    df = extract_lag_and_rolling_features(df)

    # Dropping initial rows containing NaN values from maximum lag lookback if requested.
    if drop_nas:
        max_lag = 42
        df = df.groupby(["store_id", "item_id"]).tail(-max_lag).reset_index(drop=True)

    return df
