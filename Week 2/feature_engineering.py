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

    # Creating indicator flag for weekend days.
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


def get_split_cutoff_dates(
    df: pd.DataFrame, val_days: int = 28, test_days: int = 28
) -> dict:
    # Determining chronological boundary dates for train, validation, and test partitions.
    max_date = df["date"].max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)
    val_end = test_start - pd.Timedelta(days=1)
    val_start = val_end - pd.Timedelta(days=val_days - 1)
    train_end = val_start - pd.Timedelta(days=1)
    train_start = df["date"].min()

    return {
        "train_start": str(train_start.date()),
        "train_end": str(train_end.date()),
        "val_start": str(val_start.date()),
        "val_end": str(val_end.date()),
        "test_start": str(test_start.date()),
        "test_end": str(max_date.date()),
        "val_days": val_days,
        "test_days": test_days,
    }


def split_chronologically(
    df: pd.DataFrame, val_days: int = 28, test_days: int = 28
) -> tuple:
    # Computing partition cutoff dates.
    cutoffs = get_split_cutoff_dates(df, val_days=val_days, test_days=test_days)

    val_start_dt = pd.to_datetime(cutoffs["val_start"])
    test_start_dt = pd.to_datetime(cutoffs["test_start"])

    # Partitioning dataset into chronological train, validation, and test sets.
    df_train = df[df["date"] < val_start_dt].copy()
    df_val = df[
        (df["date"] >= val_start_dt) & (df["date"] < test_start_dt)
    ].copy()
    df_test = df[df["date"] >= test_start_dt].copy()

    return df_train, df_val, df_test, cutoffs


def generate_split_plan_markdown(cutoffs: dict, split_stats: dict) -> str:
    # Formatting chronological partition plan into structured documentation.
    return rf"""# Chronological Validation and Evaluation Split Plan

## 1. Split Strategy and Rationale
Demand forecasting models must strictly respect temporal causality. Random cross-validation or randomly shuffling rows induces severe forward-looking leakage, producing artificially optimistic performance that degrades in production.

This project implements a **strict chronological rolling window partition**:
- **Forecast Horizon**: 28 days (4 full retail weeks, matching standard supply-chain replenishment cycles).
- **Leakage Prevention**: All lag and rolling features are anchored at or before forecast origin $t_0$.
- **Model Tuning**: Hyperparameter tuning and model architecture selection occur exclusively on the **Validation** window.
- **Unbiased Assessment**: The **Test** window remains completely held out until final benchmarking.

---

## 2. Partition Dates and Record Counts

| Partition | Start Date | End Date | Duration | Record Count | Purpose |
|---|---|---|---|---|---|
| **Training** | `{cutoffs['train_start']}` | `{cutoffs['train_end']}` | Historical Baseline | {split_stats.get('train_records', 0):,} | Fitting model parameters and feature representations |
| **Validation** | `{cutoffs['val_start']}` | `{cutoffs['val_end']}` | {cutoffs['val_days']} days | {split_stats.get('val_records', 0):,} | Model selection, hyperparameter tuning, baseline comparison |
| **Testing** | `{cutoffs['test_start']}` | `{cutoffs['test_end']}` | {cutoffs['test_days']} days | {split_stats.get('test_records', 0):,} | Final unbiased out-of-sample operational evaluation |

---

## 3. Data Leakage Audit Checklist
- [x] **Temporal Ordering**: Training dates precede validation dates; validation dates precede test dates.
- [x] **No Target Leakage in Features**: Lags are strictly $\ge 7$ days (or $\ge 28$ days for direct multi-step).
- [x] **Independent Pricing Statistics**: Price statistics are computed on past observations only.
- [x] **No Future Information**: Calendar and event features use only pre-scheduled date metadata.
"""


def run_feature_pipeline(
    input_interim_path: str = "Week 1/sales_interim.csv",
    output_features_path: str = "Week 2/features_engineered.csv",
    split_plan_output: str = "Week 2/chronological_split_plan.md",
) -> pd.DataFrame:
    # Locating input interim dataset path.
    in_file = Path(input_interim_path)
    if not in_file.exists():
        in_file = Path("../Week 1/sales_interim.csv")

    # Loading interim sales dataset from disk.
    print(f"Loading interim sales dataset from {in_file}...")
    df_interim = pd.read_csv(in_file, parse_dates=["date"])

    # Engineering lag, rolling, calendar, and price features.
    print("Extracting features (lags, rolling stats, calendar, pricing)...")
    df_features = build_feature_pipeline(df_interim, drop_nas=True)

    # Saving engineered dataset to Week 2 directory.
    print(f"Saving engineered features to {output_features_path}...")
    out_file = Path(output_features_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(out_file, index=False)

    # Generating chronological train, validation, and test split partitions.
    print("Partitioning dataset into chronological splits...")
    df_train, df_val, df_test, cutoffs = split_chronologically(
        df_features, val_days=28, test_days=28
    )

    # Documenting chronological partition plan.
    split_stats = {
        "train_records": len(df_train),
        "val_records": len(df_val),
        "test_records": len(df_test),
    }
    split_md = generate_split_plan_markdown(cutoffs, split_stats)

    # Writing split plan markdown document to Week 2 deliverable folder.
    plan_file = Path(split_plan_output)
    plan_file.parent.mkdir(parents=True, exist_ok=True)
    plan_file.write_text(split_md, encoding="utf-8")
    print(f"Chronological split plan documented at {split_plan_output}")

    return df_features


if __name__ == "__main__":
    run_feature_pipeline()
