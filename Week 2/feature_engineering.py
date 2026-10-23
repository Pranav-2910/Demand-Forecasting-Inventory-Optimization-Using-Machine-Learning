from pathlib import Path
import sys
import pandas as pd

# Adding project root to path for module imports.
sys.path.append(str(Path(".").resolve()))

from src.features import build_feature_pipeline
from src.splitting import split_chronologically, generate_split_plan_markdown


def run_feature_pipeline(
    input_interim_path: str = "data/interim/sales_interim.csv",
    output_features_path: str = "data/processed/features_engineered.csv",
    split_plan_output: str = "Week 2/chronological_split_plan.md",
) -> pd.DataFrame:
    # Loading interim sales dataset from disk.
    print("Loading interim sales dataset...")
    df_interim = pd.read_csv(input_interim_path, parse_dates=["date"])

    # Engineering lag, rolling, calendar, and price features.
    print("Extracting features (lags, rolling stats, calendar, pricing)...")
    df_features = build_feature_pipeline(df_interim, drop_nas=True)

    # Saving engineered dataset to processed directory.
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
