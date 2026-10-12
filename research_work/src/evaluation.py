"""
================================================================================
Evaluation Suite: Metrics, Top-k Differential Ranking, & Confusion Matrices
================================================================================
Standardized evaluation pipeline tracking:
- Accuracy, Balanced Accuracy
- Macro Precision, Macro Recall, Macro F1
- Top-3 and Top-5 Differential Diagnostic Accuracy
- Confusion Matrix generation & serialization
- Training & Inference Latency benchmarking
================================================================================
"""

import time
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from experiment_config import CONFUSION_DIR
from utilities import ensure_dir


def compute_top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int = 3) -> float:
    """Computes Top-k accuracy: whether the true label is among top k predicted probabilities."""
    if y_proba is None:
        return float("nan")
    
    # Handle binary or 1D probability output
    if len(y_proba.shape) == 1:
        return float(accuracy_score(y_true, (y_proba >= 0.5).astype(int)))
    
    # Sort descending along classes axis
    top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
    correct = np.any(top_k_preds == y_true[:, None], axis=1)
    return float(np.mean(correct))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    train_time_sec: float = 0.0,
    inference_time_sec: float = 0.0,
) -> Dict[str, Any]:
    """
    Computes full comprehensive evaluation metrics dictionary.
    """
    n_samples = len(y_true)
    latency_ms_per_sample = (inference_time_sec / max(n_samples, 1)) * 1000.0

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    
    # Use zero_division=0 to handle folds where a class might be absent in validation
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    top_3 = compute_top_k_accuracy(y_true, y_proba, k=3) if y_proba is not None else acc
    top_5 = compute_top_k_accuracy(y_true, y_proba, k=5) if y_proba is not None else acc

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "top_3_accuracy": round(top_3, 4),
        "top_5_accuracy": round(top_5, 4),
        "train_time_sec": round(train_time_sec, 4),
        "inference_time_ms_per_sample": round(latency_ms_per_sample, 4),
        "confusion_matrix": cm,
    }


def save_confusion_matrix(cm: np.ndarray, model_name: str, protocol_name: str, output_dir: Path = CONFUSION_DIR) -> Path:
    """Exports confusion matrix array to a CSV file."""
    ensure_dir(output_dir)
    clean_m = model_name.lower().replace(" ", "_")
    clean_p = protocol_name.lower().replace(" ", "_")
    cm_path = output_dir / f"cm_{clean_p}_{clean_m}.csv"
    np.savetxt(cm_path, cm, delimiter=",", fmt="%d")
    return cm_path
