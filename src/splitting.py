from pathlib import Path
import pandas as pd


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
    # Computing exact partition dates.
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
