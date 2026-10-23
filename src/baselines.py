from pathlib import Path
import numpy as np
import pandas as pd
from src.metrics import evaluate_forecasts


def generate_baseline_predictions(df: pd.DataFrame) -> dict:
    # Generating heuristic baseline predictions from lag and rolling features.
    predictions = {}

    # Generating Naive baseline predicting sales from 7 days prior.
    predictions["Naive (Lag 7)"] = df["sales_lag_7"].fillna(0).values

    # Generating Seasonal Naive baseline predicting sales from 28 days prior (same cycle day).
    predictions["Seasonal Naive (Lag 28)"] = df["sales_lag_28"].fillna(0).values

    # Generating 7-day Moving Average baseline.
    predictions["Moving Average (7-Day)"] = (
        df["rolling_mean_7"].fillna(0).values
    )

    # Generating 28-day Moving Average baseline.
    predictions["Moving Average (28-Day)"] = (
        df["rolling_mean_28"].fillna(0).values
    )

    return predictions


def evaluate_baselines_on_split(
    df_split: pd.DataFrame, split_name: str = "Validation"
) -> pd.DataFrame:
    # Evaluating all baseline forecasting heuristics on the specified partition.
    y_true = df_split["sales"].values
    baseline_preds = generate_baseline_predictions(df_split)

    results = []
    for model_name, y_pred in baseline_preds.items():
        metrics = evaluate_forecasts(
            y_true, y_pred, model_name=f"{model_name} [{split_name}]"
        )
        results.append(metrics)

    return pd.DataFrame(results)


def build_baseline_evaluation_table(
    df_val: pd.DataFrame, df_test: pd.DataFrame
) -> pd.DataFrame:
    # Evaluating baselines across both validation and testing partitions.
    val_table = evaluate_baselines_on_split(df_val, split_name="Validation")
    test_table = evaluate_baselines_on_split(df_test, split_name="Test")

    # Combining validation and testing evaluation scorecards.
    combined_table = pd.concat([val_table, test_table], ignore_index=True)
    return combined_table


def save_baseline_results(
    eval_df: pd.DataFrame,
    csv_path: str = "reports/baseline_evaluation_table.csv",
    md_path: str = "reports/baseline_evaluation_table.md",
) -> None:
    # Saving baseline evaluation results to CSV and markdown formats.
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(csv_path, index=False)

    md_lines = [
        "# Baseline Forecasting Models Evaluation Scorecard",
        "",
        "## Summary of Heuristic Baselines",
        "To establish credible benchmarks for machine learning models, four standard heuristic baselines were evaluated on identical chronological validation and test partitions.",
        "",
        "| Model Partition | MAE (Units) | RMSE | WAPE (%) | Non-Zero MAPE (%) | Forecast Bias (%) |",
        "|---|---|---|---|---|---|",
    ]

    for _, row in eval_df.iterrows():
        md_lines.append(
            f"| `{row['model']}` | {row['mae']} | {row['rmse']} | {row['wape_pct']}% | {row['mape_nonzero_pct']}% | {row['bias_pct']}% |"
        )

    md_lines.extend(
        [
            "",
            "## Key Baseline Insights",
            "- **Seasonal Naive (Lag 28)** captures monthly cyclic demand but has higher latency.",
            "- **Naive (Lag 7)** effectively captures strong weekly day-of-week seasonality (e.g. Saturday shopping peaks).",
            "- **Moving Averages** smooth noise and reduce variance, but introduce systematic lag during abrupt promotional demand spikes.",
        ]
    )

    Path(md_path).write_text("\n".join(md_lines), encoding="utf-8")
