from pathlib import Path
import sys
import pandas as pd

# Adding project root to path for module imports.
sys.path.append(str(Path(".").resolve()))

from src.splitting import split_chronologically
from src.baselines import build_baseline_evaluation_table, save_baseline_results


def run_baseline_evaluation(
    features_path: str = "data/processed/features_engineered.csv",
    csv_output_path: str = "Week 2/baseline_evaluation_table.csv",
    md_output_path: str = "Week 2/baseline_evaluation_table.md",
) -> pd.DataFrame:
    # Loading engineered features dataset from processed data store.
    print("Loading engineered features dataset...")
    df_features = pd.read_csv(features_path, parse_dates=["date"])

    # Splitting into chronological training, validation, and test sets.
    print("Partitioning chronological splits for baseline evaluation...")
    df_train, df_val, df_test, cutoffs = split_chronologically(
        df_features, val_days=28, test_days=28
    )

    # Evaluating baseline models across validation and testing partitions.
    print("Evaluating baseline models on validation and test sets...")
    eval_df = build_baseline_evaluation_table(df_val, df_test)

    # Saving evaluation table to Week 2 deliverables and reports directory.
    print(f"Saving baseline evaluation scorecard to {csv_output_path} and {md_output_path}...")
    save_baseline_results(eval_df, csv_path=csv_output_path, md_path=md_output_path)

    # Also saving a copy in reports directory for centralized reporting.
    save_baseline_results(
        eval_df,
        csv_path="reports/baseline_evaluation_table.csv",
        md_path="reports/baseline_evaluation_table.md",
    )

    print("Baseline evaluation completed successfully.")
    print(eval_df.to_string(index=False))

    return eval_df


if __name__ == "__main__":
    run_baseline_evaluation()
