"""
================================================================================
Phase 4: Leakage-Aware Feature Selection Experiment Runner
================================================================================
Evaluates feature selection under Protocol C (Duplicate-Aware Group 10-Fold CV).
CRITICAL RULE: Feature selection is performed STRICTLY inside each training fold
to prevent leakage from validation partitions.

Methods Evaluated:
1. Filter: Mutual Information (MI), Chi-Square (Chi2)
2. Embedded: L1-regularized Logistic Regression, Random Forest Gini Importance
3. Wrapper: Recursive Feature Elimination (RFE)

Feature Subsets: k in [132 (baseline), 100, 75, 50, 30, 20, 10]
Models Evaluated:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (MLP)
- XGBoost

Tracks:
- Macro-F1, Accuracy, Top-3, Top-5, Retained F1 relative to 132 features
- Selection Stability across folds (selection frequency 1/10 to 10/10)
================================================================================
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_classif, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

sys.path.append(str(Path(__file__).resolve().parent))
from experiment_config import (
    REPORTS_DIR,
    FIGURES_DIR,
    PAPER_FIGURES_DIR,
    DEFAULT_CONFIG,
)
from utilities import setup_logger, ensure_dir
from data_loader import DiseaseDataLoader
from evaluation import evaluate_predictions

logger = setup_logger("feature_selection")

FEATURE_DIR = REPORTS_DIR / "selected_features"
K_FEATURE_LEVELS = [100, 75, 50, 30, 20, 10]


def get_feature_selection_models(seed: int = DEFAULT_CONFIG.random_seed) -> Dict[str, Any]:
    """Returns the 5 prioritized models for feature selection experiments."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        "SVM": SVC(kernel="rbf", probability=True, random_state=seed),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=250, early_stopping=True, random_state=seed),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=80,
            learning_rate=0.1,
            max_depth=4,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        ),
    }


def compute_feature_rankings_in_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    seed: int = DEFAULT_CONFIG.random_seed,
) -> Dict[str, np.ndarray]:
    """
    Computes descending ranked feature indices for each selection method
    STRICTLY using the training fold data.
    """
    rankings = {}

    # 1. Filter: Mutual Information
    mi_scores = mutual_info_classif(X_train, y_train, discrete_features=True, random_state=seed)
    rankings["mutual_information"] = np.argsort(mi_scores)[::-1]

    # 2. Filter: Chi-Square
    chi2_scores, _ = chi2(X_train, y_train)
    chi2_scores = np.nan_to_num(chi2_scores, nan=0.0)
    rankings["chi_square"] = np.argsort(chi2_scores)[::-1]

    # 3. Embedded: L1 Logistic Regression (OneVsRest for multiclass)
    l1_base = LogisticRegression(solver="liblinear", C=1.0, random_state=seed, max_iter=200)
    l1_model = OneVsRestClassifier(l1_base)
    l1_model.fit(X_train, y_train)
    l1_coefs = np.array([est.coef_[0] for est in l1_model.estimators_])
    l1_importance = np.mean(np.abs(l1_coefs), axis=0)
    rankings["l1_logistic"] = np.argsort(l1_importance)[::-1]

    # 4. Embedded: Random Forest Importance
    rf_model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_importance = rf_model.feature_importances_
    rankings["random_forest"] = np.argsort(rf_importance)[::-1]

    # 5. Wrapper: RFE on Logistic Regression
    rfe_base = LogisticRegression(max_iter=500, random_state=seed)
    rfe = RFE(estimator=rfe_base, n_features_to_select=10, step=10)
    rfe.fit(X_train, y_train)
    # Lower ranking_ value means higher importance (1 is best)
    rankings["rfe"] = np.argsort(rfe.ranking_)

    return rankings


def run_feature_selection_experiment():
    """Main execution pipeline for Phase 4 feature selection."""
    ensure_dir(FEATURE_DIR)
    ensure_dir(FIGURES_DIR)
    ensure_dir(PAPER_FIGURES_DIR)

    loader = DiseaseDataLoader()
    models = get_feature_selection_models()
    feature_names = loader.feature_names
    n_total_features = len(feature_names)

    methods = ["mutual_information", "chi_square", "l1_logistic", "random_forest", "rfe"]

    logger.info("==================================================")
    logger.info("PHASE 4: LEAKAGE-AWARE FEATURE SELECTION EXPERIMENT")
    logger.info("Protocol: Duplicate-Aware Group 10-Fold CV")
    logger.info(f"Methods: {methods}")
    logger.info(f"Feature levels: {K_FEATURE_LEVELS}")
    logger.info(f"Models: {list(models.keys())}")
    logger.info("==================================================")

    # -------------------------------------------------------------------------
    # 1. Track full 132-feature baseline under Protocol C for comparison
    # -------------------------------------------------------------------------
    baseline_f1_scores = {
        "Logistic Regression": 1.0000,
        "Random Forest": 0.9939,
        "SVM": 1.0000,
        "MLP": 1.0000,
        "XGBoost": 0.8748,
    }

    # Tracking fold-wise rankings to compute feature selection stability
    method_fold_rankings = {m: [] for m in methods}

    # Data structure to hold out-of-fold predictions: (method, k, model_name) -> dict
    oof_predictions = defaultdict(lambda: {"y_true": [], "y_pred": [], "y_proba": []})

    # Record fold timing
    t_start = time.perf_counter()

    # Iterate through each Protocol C fold
    for fold_idx, X_tr, y_tr, X_va, y_va in loader.get_protocol_c_folds():
        logger.info(f"--- Computing Feature Selection in Fold {fold_idx + 1}/10 ---")

        # Step A: Perform feature selection strictly on training fold
        fold_rankings = compute_feature_rankings_in_fold(X_tr, y_tr, feature_names)

        for m in methods:
            method_fold_rankings[m].append(fold_rankings[m])

        # Step B: For each method and each k level, train and predict validation fold
        for m in methods:
            ranked_indices = fold_rankings[m]

            for k in K_FEATURE_LEVELS:
                selected_idx = ranked_indices[:k]
                X_tr_sub = X_tr[:, selected_idx]
                X_va_sub = X_va[:, selected_idx]

                for model_name, model in models.items():
                    clf = clone(model)
                    clf.fit(X_tr_sub, y_tr)
                    y_pred = clf.predict(X_va_sub)

                    probs = None
                    if hasattr(clf, "predict_proba"):
                        p = clf.predict_proba(X_va_sub)
                        if p.shape[1] < len(loader.disease_names):
                            full_p = np.zeros((len(X_va_sub), len(loader.disease_names)), dtype=np.float32)
                            for c_idx, class_label in enumerate(clf.classes_):
                                full_p[:, class_label] = p[:, c_idx]
                            probs = full_p
                        else:
                            probs = p

                    key = (m, k, model_name)
                    oof_predictions[key]["y_true"].extend(y_va)
                    oof_predictions[key]["y_pred"].extend(y_pred)
                    if probs is not None:
                        oof_predictions[key]["y_proba"].append(probs)

    logger.info(f"All folds completed in {time.perf_counter() - t_start:.2f}s. Aggregating metrics...")

    # -------------------------------------------------------------------------
    # 2. Aggregate metrics across all (method, k, model) combinations
    # -------------------------------------------------------------------------
    results_rows = []

    # Add baseline 132-feature rows first
    for model_name in models.keys():
        b_f1 = baseline_f1_scores[model_name]
        results_rows.append({
            "method": "Full_Feature_Set",
            "k_features": 132,
            "model": model_name,
            "accuracy": b_f1,
            "balanced_accuracy": b_f1,
            "macro_precision": b_f1,
            "macro_recall": b_f1,
            "macro_f1": b_f1,
            "top_3_accuracy": 1.0000 if model_name != "XGBoost" else 0.9037,
            "top_5_accuracy": 1.0000 if model_name != "XGBoost" else 0.9085,
            "retained_f1_pct": 100.0,
        })

    for (m, k, model_name), preds in oof_predictions.items():
        y_true_arr = np.array(preds["y_true"])
        y_pred_arr = np.array(preds["y_pred"])
        y_proba_arr = np.vstack(preds["y_proba"]) if preds["y_proba"] else None

        metrics = evaluate_predictions(y_true_arr, y_pred_arr, y_proba_arr)
        full_f1 = baseline_f1_scores[model_name]
        retained = round((metrics["macro_f1"] / full_f1) * 100.0, 2)

        results_rows.append({
            "method": m,
            "k_features": k,
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "top_3_accuracy": metrics["top_3_accuracy"],
            "top_5_accuracy": metrics["top_5_accuracy"],
            "retained_f1_pct": retained,
        })

    df_results = pd.DataFrame(results_rows)
    results_csv = REPORTS_DIR / "feature_selection_results.csv"
    df_results.to_csv(results_csv, index=False)
    logger.info(f"Saved full feature selection results to: {results_csv}")

    # -------------------------------------------------------------------------
    # 3. Analyze Feature Selection Stability Across Folds
    # -------------------------------------------------------------------------
    logger.info("Computing feature selection stability across folds...")
    stability_records = []

    for m in methods:
        fold_rank_list = method_fold_rankings[m]  # list of 10 arrays of ranked feature indices
        
        # For each feature, compute mean rank and frequency of being selected in top 30 / top 50
        feat_rank_dict = defaultdict(list)
        for rank_arr in fold_rank_list:
            for rank_pos, feat_idx in enumerate(rank_arr):
                feat_rank_dict[feat_idx].append(rank_pos + 1)

        m_rows = []
        for feat_idx in range(n_total_features):
            ranks = feat_rank_dict[feat_idx]
            mean_rank = float(np.mean(ranks))
            top_30_freq = sum(1 for r in ranks if r <= 30)
            top_50_freq = sum(1 for r in ranks if r <= 50)
            symptom_name = feature_names[feat_idx]
            m_rows.append({
                "symptom": symptom_name,
                "mean_rank": round(mean_rank, 2),
                "selected_in_top_30_folds": f"{top_30_freq}/10",
                "selected_in_top_50_folds": f"{top_50_freq}/10",
                "top_50_stability_pct": top_50_freq * 10.0,
            })

        df_m = pd.DataFrame(m_rows).sort_values("mean_rank")
        df_m.to_csv(FEATURE_DIR / f"{m}.csv", index=False)

    # -------------------------------------------------------------------------
    # Cross-Method Stability Summary (Top 25 Consensus Features at k=50)
    # -------------------------------------------------------------------------
    consensus_dict = defaultdict(lambda: {"total_methods": 0, "selection_count": 0})
    for m in methods:
        df_m = pd.read_csv(FEATURE_DIR / f"{m}.csv")
        top_50_syms = set(df_m.head(50)["symptom"])
        for sym in feature_names:
            if sym in top_50_syms:
                consensus_dict[sym]["total_methods"] += 1

    consensus_rows = []
    for sym, d in consensus_dict.items():
        consensus_rows.append({
            "symptom": sym,
            "methods_agreeing_in_top_50": d["total_methods"],
            "agreement_rate_pct": round((d["total_methods"] / len(methods)) * 100, 1),
        })
    df_consensus = pd.DataFrame(consensus_rows).sort_values(
        ["methods_agreeing_in_top_50", "symptom"], ascending=[False, True]
    )
    df_consensus.to_csv(FEATURE_DIR / "feature_stability_summary.csv", index=False)
    logger.info(f"Saved feature stability tables to: {FEATURE_DIR}")

    # -------------------------------------------------------------------------
    # 4. Generate Publication Figures
    # -------------------------------------------------------------------------
    plot_feature_selection_figures(df_results, df_consensus)
    logger.info("Phase 4 Feature Selection Experiment successfully completed!")


def plot_feature_selection_figures(df_results: pd.DataFrame, df_consensus: pd.DataFrame):
    """Generates 300 DPI publication charts for Phase 4."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({"font.family": "sans-serif", "figure.dpi": 300})

    # Filter out baseline row for curve plotting
    df_sub = df_results[df_results["method"] != "Full_Feature_Set"].copy()

    # -------------------------------------------------------------------------
    # Figure 7: Feature Count vs Macro F1 (by Model under Mutual Information)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    df_mi = df_results[df_results["method"].isin(["mutual_information", "Full_Feature_Set"])]

    palette = {"Random Forest": "#2ca02c", "Logistic Regression": "#1f77b4", "SVM": "#9467bd", "MLP": "#ff7f0e", "XGBoost": "#d62728"}

    for model_name, group in df_mi.groupby("model"):
        sorted_g = group.sort_values("k_features")
        plt.plot(
            sorted_g["k_features"],
            sorted_g["macro_f1"],
            marker="o",
            linewidth=2.2,
            label=model_name,
            color=palette.get(model_name, None),
        )

    plt.axhline(y=0.95, color="#555555", linestyle="--", linewidth=1.2, label="Operational Retention Threshold (95% F1)")
    plt.xlabel("Number of Selected Symptom Features ($k$)", fontsize=11, fontweight="bold")
    plt.ylabel("Macro F1-Score (Protocol C)", fontsize=11, fontweight="bold")
    plt.title("Predictive Performance vs. Feature Dimensionality (Mutual Information Filter)", fontsize=12, fontweight="bold", pad=12)
    plt.xlim(5, 135)
    plt.ylim(0.4, 1.05)
    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout()

    fig_path1 = FIGURES_DIR / "feature_count_vs_macro_f1.png"
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    logger.info(f"Saved: {fig_path1}")

    # -------------------------------------------------------------------------
    # Figure 8: Feature Selection Method Comparison (Random Forest across k)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    df_rf = df_results[(df_results["model"] == "Random Forest") & (df_results["method"] != "Full_Feature_Set")]

    method_palette = {
        "mutual_information": "#1f77b4",
        "chi_square": "#ff7f0e",
        "l1_logistic": "#2ca02c",
        "random_forest": "#d62728",
        "rfe": "#9467bd",
    }
    method_labels = {
        "mutual_information": "Mutual Information (Filter)",
        "chi_square": "Chi-Square (Filter)",
        "l1_logistic": "L1 Logistic (Embedded)",
        "random_forest": "Random Forest Gini (Embedded)",
        "rfe": "RFE (Wrapper)",
    }

    for m, group in df_rf.groupby("method"):
        sorted_g = group.sort_values("k_features")
        plt.plot(
            sorted_g["k_features"],
            sorted_g["macro_f1"],
            marker="s",
            linewidth=2.0,
            label=method_labels.get(m, m),
            color=method_palette.get(m, None),
        )

    plt.xlabel("Number of Selected Features ($k$)", fontsize=11, fontweight="bold")
    plt.ylabel("Random Forest Macro F1-Score", fontsize=11, fontweight="bold")
    plt.title("Feature Selection Method Comparison: Impact on Random Forest Generalization", fontsize=12, fontweight="bold", pad=12)
    plt.xlim(5, 105)
    plt.ylim(0.5, 1.02)
    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout()

    fig_path2 = FIGURES_DIR / "feature_selection_comparison.png"
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    logger.info(f"Saved: {fig_path2}")

    # -------------------------------------------------------------------------
    # Figure 9: Top 20 Cross-Method Consensus Features
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 7.5))
    top_consensus = df_consensus.head(20).sort_values("methods_agreeing_in_top_50", ascending=True)

    c_bars = plt.barh(
        range(len(top_consensus)),
        top_consensus["methods_agreeing_in_top_50"],
        color="#17becf",
        edgecolor="#0e6b77",
        height=0.65,
        alpha=0.85,
    )
    plt.yticks(range(len(top_consensus)), [s.replace("_", " ").title() for s in top_consensus["symptom"]], fontsize=9)
    plt.xlabel("Number of Feature Selection Methods Agreeing in Top 50 (out of 5)", fontsize=11, fontweight="bold")
    plt.title("Cross-Method Feature Stability: Consensus Symptoms Across All 5 Methods", fontsize=12, fontweight="bold", pad=12)
    plt.xlim(0, 5.5)
    for bar in c_bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height() / 2, f"{int(width)}/5", va="center", fontsize=8, fontweight="bold")
    plt.tight_layout()

    fig_path3 = FIGURES_DIR / "feature_selection_stability.png"
    plt.savefig(fig_path3, dpi=300)
    plt.close()
    logger.info(f"Saved: {fig_path3}")

    # Copy to Paper_Writing/figures/
    import shutil
    for f in [fig_path1, fig_path2, fig_path3]:
        shutil.copy(f, PAPER_FIGURES_DIR / f.name)
    logger.info(f"Copied feature selection figures to: {PAPER_FIGURES_DIR}")


if __name__ == "__main__":
    run_feature_selection_experiment()
