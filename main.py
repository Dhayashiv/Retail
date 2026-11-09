"""Run the complete project:  python main.py   (optionally:  python main.py --data your_file.csv)"""
import argparse
from pathlib import Path

import pandas as pd

from src.generate_data import generate_retail_data
from src.preprocess import load_and_clean
from src.eda import run_eda
from src.model import train_and_evaluate
from src.report import write_report

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Retail Sales Analysis & Demand Prediction")
    parser.add_argument("--data", default=str(ROOT / "data" / "retail_sales.csv"),
                        help="CSV path (generated automatically if it does not exist)")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = ROOT / "outputs"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(exist_ok=True)

    if not data_path.exists():
        print("[1/5] Dataset not found - generating synthetic retail dataset ...")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        generate_retail_data().to_csv(data_path, index=False)
    else:
        print(f"[1/5] Using dataset: {data_path}")

    print("[2/5] Loading and cleaning data ...")
    df = load_and_clean(str(data_path))
    df.to_csv(out_dir / "cleaned_data.csv", index=False)
    print(f"      {len(df):,} rows, {df['date'].min().date()} to {df['date'].max().date()}")

    print("[3/5] Running exploratory analysis and saving charts ...")
    insights = run_eda(df, fig_dir)

    print("[4/5] Training and evaluating models (this takes ~1 minute) ...")
    results, best_name, importance = train_and_evaluate(df, fig_dir, out_dir / "models")
    results.to_csv(out_dir / "model_results.csv", index=False)
    print(results.to_string(index=False))

    print("[5/5] Writing report ...")
    write_report(insights, results, best_name, importance, out_dir / "report.md")

    print("\nDone. Open the 'outputs' folder:")
    print("  outputs/report.md        - findings and conclusions")
    print("  outputs/figures/         - all visualisations")
    print("  outputs/model_results.csv, outputs/cleaned_data.csv, outputs/models/best_model.joblib")


if __name__ == "__main__":
    main()
