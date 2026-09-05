"""Train XGBoost Cost Model on Measured Empirical Benchmark Dataset."""

import os
import math
import pickle
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
from rich.console import Console
from rich.table import Table

from markovlens.model import FEATURE_NAMES, ProfitabilityModel

console = Console()

def train_and_evaluate(dataset_csv: str = "dataset.csv", output_model: str = "markovlens/trained_model.pkl"):
    if not os.path.exists(dataset_csv):
        console.print(f"[bold red]Error: Dataset file {dataset_csv} not found. Run scripts/collect_data.py first.[/bold red]")
        return

    console.print(f"[bold cyan]MarkovLens Model Training & Cross-Validation[/bold cyan]")
    df = pd.read_csv(dataset_csv)
    console.print(f"Loaded [green]{len(df)}[/green] benchmark rows from [yellow]{dataset_csv}[/yellow]\n")

    # Target: log2(Speedup) for numerical stability and symmetric penalty
    df["target_log_speedup"] = np.log2(np.maximum(df["speedup"].values, 0.001))

    X = df[FEATURE_NAMES]
    y = df["target_log_speedup"]
    y_class_true = df["is_profitable"].values

    # 5-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = []
    rmse_scores = []
    accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        y_class_val = y_class_true[val_idx]

        model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        pred_speedup = 2.0 ** y_pred
        y_pred_class = (pred_speedup >= 1.1).astype(int)

        r2_scores.append(r2_score(y_val, y_pred))
        rmse_scores.append(math.sqrt(mean_squared_error(y_val, y_pred)))
        accuracies.append(accuracy_score(y_class_val, y_pred_class))

    # Print Validation Metrics
    metrics_table = Table(title="5-Fold Cross-Validation Performance")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Score (Mean +/- Std)", style="green", justify="right")

    metrics_table.add_row("Regression R² Score", f"{np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    metrics_table.add_row("Log-Speedup RMSE", f"{np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    metrics_table.add_row("Profitability Gating Accuracy", f"{np.mean(accuracies) * 100:.2f}% ± {np.std(accuracies) * 100:.2f}%")

    console.print(metrics_table)

    # Train final full model
    console.print("\n[yellow]Training final model on 100% of dataset...[/yellow]")
    final_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    final_model.fit(X, y)

    # Save artifact
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    with open(output_model, 'wb') as f:
        pickle.dump({"model": final_model}, f)

    console.print(f"[bold green][SUCCESS] Model artifact saved successfully to {output_model}[/bold green]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MarkovLens XGBoost Cost Model.")
    parser.add_argument("--data", default="dataset.csv", help="Input dataset CSV.")
    parser.add_argument("--out", default="markovlens/trained_model.pkl", help="Output model pickle path.")
    args = parser.parse_args()

    train_and_evaluate(args.data, args.out)
