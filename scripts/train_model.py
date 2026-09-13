import sys
import os
import math
import pickle
import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from rich.console import Console
from rich.table import Table

from cerberus.model import FEATURE_NAMES, FEATURE_LABELS, MONOTONIC_CONSTRAINTS, engineer_features

console = Console()

def train_and_evaluate(dataset_csv: str = "dataset/dataset_merged.csv", output_model: str = "cerberus/trained_model.pkl"):
    if not os.path.exists(dataset_csv):
        console.print(f"[bold red]Error: Dataset file {dataset_csv} not found.[/bold red]")
        return

    console.print(f"[bold cyan]=== Cerberus Two-Stage Hurdle Training Pipeline ===[/bold cyan]")
    df_raw = pd.read_csv(dataset_csv)
    
    # 1. Data Sanitization
    df = df_raw.drop_duplicates().dropna()
    console.print(f"Loaded [green]{len(df)}[/green] clean benchmark rows from [yellow]{dataset_csv}[/yellow]")
    
    pos_count = int(df["is_profitable"].sum())
    neg_count = len(df) - pos_count
    console.print(f"Class Balance: [green]{pos_count} Profitable ({pos_count/len(df):.1%})[/green] | [red]{neg_count} Unprofitable ({neg_count/len(df):.1%})[/red]")
    
    # 2. Target Variable Calibration: Log2 speedup with lower-tail saturation at 0.05x
    clamped_speedup = np.maximum(df["speedup"].values, 0.05)
    df["target_log_speedup"] = np.log2(clamped_speedup)

    # 3. Domain-Specific Feature Engineering
    X = engineer_features(df)
    y = df["target_log_speedup"]
    y_class_true = df["is_profitable"].values

    console.print(f"Engineered Feature Vector: [cyan]{len(FEATURE_NAMES)} features[/cyan] with Physics Monotonic Constraints\n")

    # 4. Stratified 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    roc_aucs = []
    igpu_accs = []
    dgpu_accs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_class_true)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        y_class_train, y_class_val = y_class_true[train_idx], y_class_true[val_idx]
        hw_val = df["unified_memory"].iloc[val_idx].values

        # Stage 1: Classifier (Gating)
        clf = xgb.XGBClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.06,
            monotone_constraints=MONOTONIC_CONSTRAINTS,
            random_state=42
        )
        clf.fit(X_train, y_class_train)
        y_prob = clf.predict_proba(X_val)[:, 1]

        # Stage 2: Regressor (Magnitude)
        reg = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.06,
            subsample=0.88,
            colsample_bytree=0.88,
            reg_alpha=0.05,
            reg_lambda=1.0,
            monotone_constraints=MONOTONIC_CONSTRAINTS,
            random_state=42
        )
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_val)

        # Combined Two-Stage Decision
        pred_speedup = 2.0 ** y_pred
        y_pred_class = ((y_prob >= 0.5) & (pred_speedup >= 1.05)).astype(int)

        r2_scores.append(r2_score(y_val, y_pred))
        rmse_scores.append(math.sqrt(mean_squared_error(y_val, y_pred)))
        mae_scores.append(mean_absolute_error(y_val, y_pred))
        accuracies.append(accuracy_score(y_class_val, y_pred_class))
        precisions.append(precision_score(y_class_val, y_pred_class, zero_division=0))
        recalls.append(recall_score(y_class_val, y_pred_class, zero_division=0))
        f1s.append(f1_score(y_class_val, y_pred_class, zero_division=0))
        roc_aucs.append(roc_auc_score(y_class_val, y_prob))

        # Subgroup Accuracies (iGPU vs dGPU)
        igpu_mask = (hw_val == 1.0)
        dgpu_mask = (hw_val == 0.0)
        if igpu_mask.any():
            igpu_accs.append(accuracy_score(y_class_val[igpu_mask], y_pred_class[igpu_mask]))
        if dgpu_mask.any():
            dgpu_accs.append(accuracy_score(y_class_val[dgpu_mask], y_pred_class[dgpu_mask]))

    # Print Validation Metrics
    metrics_table = Table(title="Two-Stage Hurdle Model 5-Fold Cross-Validation Performance")
    metrics_table.add_column("Evaluation Metric", style="cyan")
    metrics_table.add_column("Score (Mean +/- Std)", style="green", justify="right")

    metrics_table.add_row("Classification ROC-AUC", f"{np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")
    metrics_table.add_row("Offload Gating Accuracy", f"{np.mean(accuracies) * 100:.2f}% ± {np.std(accuracies) * 100:.2f}%")
    metrics_table.add_row("Offload Decision Precision", f"{np.mean(precisions) * 100:.2f}% ± {np.std(precisions) * 100:.2f}%")
    metrics_table.add_row("Offload Decision Recall", f"{np.mean(recalls) * 100:.2f}% ± {np.std(recalls) * 100:.2f}%")
    metrics_table.add_row("Offload F1-Score", f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    metrics_table.add_row("Regression R² Score", f"{np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    metrics_table.add_row("Log2-Speedup RMSE", f"{np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    metrics_table.add_row("Log2-Speedup MAE", f"{np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    metrics_table.add_row("iGPU Subgroup Accuracy", f"{np.mean(igpu_accs) * 100:.2f}% ± {np.std(igpu_accs) * 100:.2f}%")
    metrics_table.add_row("dGPU Subgroup Accuracy", f"{np.mean(dgpu_accs) * 100:.2f}% ± {np.std(dgpu_accs) * 100:.2f}%")

    console.print(metrics_table)

    # 5. Train final model on 100% of dataset
    console.print("\n[yellow]Training production Two-Stage Hurdle model on full merged dataset...[/yellow]")
    final_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.06,
        monotone_constraints=MONOTONIC_CONSTRAINTS,
        random_state=42
    )
    final_clf.fit(X, y_class_true)

    final_reg = xgb.XGBRegressor(
        n_estimators=220,
        max_depth=6,
        learning_rate=0.055,
        subsample=0.88,
        colsample_bytree=0.88,
        reg_alpha=0.05,
        reg_lambda=1.0,
        monotone_constraints=MONOTONIC_CONSTRAINTS,
        random_state=42
    )
    final_reg.fit(X, y)

    # Print Top Feature Importances
    imp_table = Table(title="Top Feature Importances (Gini Gain)")
    imp_table.add_column("Rank", justify="center", style="dim")
    imp_table.add_column("Feature", style="cyan")
    imp_table.add_column("Importance Weight", style="green", justify="right")

    importances = final_reg.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for rank, idx in enumerate(sorted_idx[:8], 1):
        feat_name = FEATURE_NAMES[idx]
        label = FEATURE_LABELS.get(feat_name, feat_name)
        imp_table.add_row(str(rank), label, f"{importances[idx]:.4f}")

    console.print(imp_table)

    # Save artifact with dynamic model metadata
    metadata = {
        "n_samples": len(df),
        "roc_auc": round(float(np.mean(roc_aucs)), 3),
        "roc_auc_std": round(float(np.std(roc_aucs)), 3),
        "r2": round(float(np.mean(r2_scores)), 3),
        "accuracy": round(float(np.mean(accuracies)), 3),
        "precision": round(float(np.mean(precisions)), 3),
        "recall": round(float(np.mean(recalls)), 3),
        "f1": round(float(np.mean(f1s)), 3),
        "rmse": round(float(np.mean(rmse_scores)), 3),
        "mae": round(float(np.mean(mae_scores)), 3),
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES)
    }

    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    with open(output_model, 'wb') as f:
        pickle.dump({
            "classifier": final_clf,
            "regressor": final_reg,
            "model": final_reg,
            "metadata": metadata
        }, f)

    console.print(f"\n[bold green][SUCCESS] Production Two-Stage Hurdle model exported to {output_model} (ROC-AUC: {metadata['roc_auc']}, N={metadata['n_samples']})[/bold green]")

if __name__ == "__main__":
    default_dataset = "dataset/dataset_merged.csv" if os.path.exists("dataset/dataset_merged.csv") else ("dataset/dataset.csv" if os.path.exists("dataset/dataset.csv") else "dataset.csv")
    parser = argparse.ArgumentParser(description="Train Cerberus XGBoost Cost Model.")
    parser.add_argument("--data", default=default_dataset, help="Input dataset CSV.")
    parser.add_argument("--out", default="cerberus/trained_model.pkl", help="Output model pickle path.")
    args = parser.parse_args()

    train_and_evaluate(args.data, args.out)

