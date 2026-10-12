"""
================================================================================
Phase 3: Multi-Protocol Baseline Benchmarking Runner
================================================================================
Executes the structured 9-model hierarchy across three evaluation protocols:
- Protocol A: Original Kaggle Split (4,920 -> 42)
- Protocol B: Standard Stratified 10-Fold CV
- Protocol C: Duplicate-Aware Group 10-Fold CV (Zero Pattern Leakage)

Exports:
- research/reports/baseline_results.csv
- research/reports/protocol_comparison.csv
- research/reports/confusion_matrices/
- research/reports/figures/protocol_comparison.png
- research/reports/figures/model_performance.png
================================================================================
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone

sys.path.append(str(Path(__file__).resolve().parent))
from experiment_config import (
    REPORTS_DIR,
    FIGURES_DIR,
    CONFUSION_DIR,
    PAPER_FIGURES_DIR,
    Protocol,
    DEFAULT_CONFIG,
)
from utilities import setup_logger, ensure_dir
from data_loader import DiseaseDataLoader
from models import get_model_registry
from evaluation import evaluate_predictions, save_confusion_matrix

logger = setup_logger("run_benchmarks")


def run_protocol_a(loader: DiseaseDataLoader, model_registry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Runs Protocol A: Original Kaggle Split."""
    logger.info("==================================================")
    logger.info("RUNNING PROTOCOL A: ORIGINAL KAGGLE SPLIT")
    logger.info("==================================================")
    X_train, y_train, X_test, y_test = loader.get_protocol_a_data()
    results = []

    for model_name, model in model_registry.items():
        logger.info(f"[Protocol A] Training {model_name}...")
        clf = clone(model)

        # Time training
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        t_train = time.perf_counter() - t0

        # Time inference
        t1 = time.perf_counter()
        y_pred = clf.predict(X_test)
        t_inf = time.perf_counter() - t1

        y_proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None

        metrics = evaluate_predictions(y_test, y_pred, y_proba, t_train, t_inf)
        save_confusion_matrix(metrics["confusion_matrix"], model_name, Protocol.A_KAGGLE)

        res_row = {
            "protocol": Protocol.A_KAGGLE,
            "model": model_name,
            **{k: v for k, v in metrics.items() if k != "confusion_matrix"}
        }
        results.append(res_row)
        logger.info(f" -> {model_name}: Acc={metrics['accuracy']:.4f}, Macro-F1={metrics['macro_f1']:.4f}, "
                    f"Top-3={metrics['top_3_accuracy']:.4f}, TrainTime={metrics['train_time_sec']:.2f}s")

    return results


def run_kfold_protocol(loader: DiseaseDataLoader, model_registry: Dict[str, Any], protocol_name: str) -> List[Dict[str, Any]]:
    """Runs cross-validation under Protocol B (Stratified CV) or Protocol C (Group CV)."""
    logger.info("==================================================")
    logger.info(f"RUNNING {protocol_name.upper()}")
    logger.info("==================================================")

    results = []

    for model_name, model in model_registry.items():
        logger.info(f"[{protocol_name}] Evaluating {model_name} across 10 folds...")
        fold_metrics = []
        total_train_time = 0.0
        total_inf_time = 0.0

        all_y_true = []
        all_y_pred = []
        all_y_proba = []

        # Obtain fold generator
        if protocol_name == Protocol.B_STRATIFIED:
            fold_gen = loader.get_protocol_b_folds()
        else:
            fold_gen = loader.get_protocol_c_folds()

        for fold_idx, X_train, y_train, X_val, y_val in fold_gen:
            clf = clone(model)

            # Train
            t0 = time.perf_counter()
            clf.fit(X_train, y_train)
            t_tr = time.perf_counter() - t0
            total_train_time += t_tr

            # Predict
            t1 = time.perf_counter()
            y_pred = clf.predict(X_val)
            t_inf = time.perf_counter() - t1
            total_inf_time += t_inf

            # Probabilities (handle classes that might be absent in a single validation fold)
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X_val)
                # If model classes don't cover all 41 classes (rare edge case in small fold)
                if probs.shape[1] < len(loader.disease_names):
                    full_probs = np.zeros((len(X_val), len(loader.disease_names)), dtype=np.float32)
                    for c_idx, class_label in enumerate(clf.classes_):
                        full_probs[:, class_label] = probs[:, c_idx]
                    probs = full_probs
            else:
                probs = None

            all_y_true.extend(y_val)
            all_y_pred.extend(y_pred)
            if probs is not None:
                all_y_proba.append(probs)

        # Aggregate overall out-of-fold predictions
        y_true_arr = np.array(all_y_true)
        y_pred_arr = np.array(all_y_pred)
        y_proba_arr = np.vstack(all_y_proba) if all_y_proba else None

        overall_metrics = evaluate_predictions(
            y_true_arr,
            y_pred_arr,
            y_proba_arr,
            train_time_sec=total_train_time,
            inference_time_sec=total_inf_time,
        )

        save_confusion_matrix(overall_metrics["confusion_matrix"], model_name, protocol_name)

        res_row = {
            "protocol": protocol_name,
            "model": model_name,
            **{k: v for k, v in overall_metrics.items() if k != "confusion_matrix"}
        }
        results.append(res_row)
        logger.info(f" -> {model_name} Overall: Acc={overall_metrics['accuracy']:.4f}, "
                    f"Macro-F1={overall_metrics['macro_f1']:.4f}, Top-3={overall_metrics['top_3_accuracy']:.4f}")

    return results


def plot_benchmark_figures(df_results: pd.DataFrame) -> None:
    """Generates publication-quality charts for the baseline benchmarks."""
    ensure_dir(FIGURES_DIR)
    ensure_dir(PAPER_FIGURES_DIR)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "figure.dpi": 300,
    })

    # -------------------------------------------------------------------------
    # Figure 5: Protocol Comparison (Macro F1 across Protocols A, B, C)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    pivoted = df_results.pivot(index="model", columns="protocol", values="macro_f1")
    
    # Reorder models logically
    model_order = [
        "Dummy Classifier",
        "Naive Bayes",
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Extra Trees",
        "SVM",
        "XGBoost",
        "MLP",
    ]
    pivoted = pivoted.reindex([m for m in model_order if m in pivoted.index])

    protocol_cols = [Protocol.A_KAGGLE, Protocol.B_STRATIFIED, Protocol.C_GROUP]
    pivoted = pivoted[[c for c in protocol_cols if c in pivoted.columns]]

    labels = {
        Protocol.A_KAGGLE: "Protocol A (Original Kaggle Split)",
        Protocol.B_STRATIFIED: "Protocol B (Standard Stratified 10-Fold CV)",
        Protocol.C_GROUP: "Protocol C (Duplicate-Aware Group 10-Fold CV)",
    }
    pivoted.columns = [labels.get(c, c) for c in pivoted.columns]

    ax = pivoted.plot(
        kind="bar",
        figsize=(12, 6.5),
        colormap="viridis",
        edgecolor="#222222",
        width=0.8,
        alpha=0.9,
    )
    plt.ylabel("Macro F1-Score", fontsize=11, fontweight="bold")
    plt.xlabel("Machine Learning Algorithm", fontsize=11, fontweight="bold")
    plt.title("Generalization Under Leakage Removal: Benchmarking Across Protocols A, B, and C", fontsize=12, fontweight="bold", pad=12)
    plt.xticks(rotation=25, ha="right", fontsize=9, fontweight="bold")
    plt.ylim(0, 1.15)
    plt.legend(frameon=True, loc="upper left")
    plt.tight_layout()

    fig5_path = FIGURES_DIR / "protocol_comparison.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {fig5_path}")

    # -------------------------------------------------------------------------
    # Figure 6: Model Performance Under Protocol C (Duplicate-Aware CV)
    # -------------------------------------------------------------------------
    df_c = df_results[df_results["protocol"] == Protocol.C_GROUP].set_index("model")
    df_c = df_c.reindex([m for m in model_order if m in df_c.index])

    metric_cols = ["accuracy", "balanced_accuracy", "macro_f1", "top_3_accuracy", "top_5_accuracy"]
    df_c_plot = df_c[metric_cols]
    df_c_plot.columns = ["Accuracy", "Balanced Acc", "Macro F1", "Top-3 Acc", "Top-5 Acc"]

    plt.figure(figsize=(12, 6.5))
    ax2 = df_c_plot.plot(
        kind="bar",
        figsize=(13, 6.5),
        colormap="Spectral",
        edgecolor="#222222",
        width=0.8,
        alpha=0.9,
    )
    plt.ylabel("Performance Score", fontsize=11, fontweight="bold")
    plt.xlabel("Machine Learning Algorithm", fontsize=11, fontweight="bold")
    plt.title("Clinical Model Performance Under Protocol C (Zero Pattern Leakage)", fontsize=12, fontweight="bold", pad=12)
    plt.xticks(rotation=25, ha="right", fontsize=9, fontweight="bold")
    plt.ylim(0, 1.15)
    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout()

    fig6_path = FIGURES_DIR / "model_performance.png"
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {fig6_path}")

    # Copy to Paper_Writing/figures/
    import shutil
    for f in [fig5_path, fig6_path]:
        shutil.copy(f, PAPER_FIGURES_DIR / f.name)
    logger.info(f"Copied benchmark figures to {PAPER_FIGURES_DIR}")


def main():
    loader = DiseaseDataLoader()
    models = get_model_registry()

    all_results = []

    # 1. Run Protocol A
    res_a = run_protocol_a(loader, models)
    all_results.extend(res_a)

    # 2. Run Protocol B
    res_b = run_kfold_protocol(loader, models, Protocol.B_STRATIFIED)
    all_results.extend(res_b)

    # 3. Run Protocol C
    res_c = run_kfold_protocol(loader, models, Protocol.C_GROUP)
    all_results.extend(res_c)

    # Convert to DataFrame
    df_all = pd.DataFrame(all_results)
    
    # Save baseline_results.csv
    baseline_csv = REPORTS_DIR / "baseline_results.csv"
    df_all.to_csv(baseline_csv, index=False)
    logger.info(f"Saved all results to {baseline_csv}")

    # Pivot table for Protocol Comparison
    pivoted_f1 = df_all.pivot(index="model", columns="protocol", values="macro_f1")
    pivoted_acc = df_all.pivot(index="model", columns="protocol", values="accuracy")
    pivoted_top3 = df_all.pivot(index="model", columns="protocol", values="top_3_accuracy")
    
    comp_df = pd.DataFrame({
        "Model": pivoted_f1.index,
        "Protocol_A_Accuracy": pivoted_acc[Protocol.A_KAGGLE].values,
        "Protocol_A_Macro_F1": pivoted_f1[Protocol.A_KAGGLE].values,
        "Protocol_B_Accuracy": pivoted_acc[Protocol.B_STRATIFIED].values,
        "Protocol_B_Macro_F1": pivoted_f1[Protocol.B_STRATIFIED].values,
        "Protocol_C_Accuracy": pivoted_acc[Protocol.C_GROUP].values,
        "Protocol_C_Macro_F1": pivoted_f1[Protocol.C_GROUP].values,
        "Protocol_C_Top3_Acc": pivoted_top3[Protocol.C_GROUP].values,
    })
    comp_csv = REPORTS_DIR / "protocol_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    logger.info(f"Saved protocol comparison to {comp_csv}")

    # Generate Figures
    plot_benchmark_figures(df_all)
    logger.info("Phase 3 Multi-Protocol Baseline Benchmarks successfully completed!")


if __name__ == "__main__":
    main()
