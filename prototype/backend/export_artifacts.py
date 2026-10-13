"""
================================================================================
Phase 09: Model Freezing and Packaging Script
================================================================================
Freezes the verified Random Forest model, nested Isotonic calibrator,
feature catalog, canonical disease mappings, and metadata into prototype/artifacts.
================================================================================
"""

import sys
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
import shap

# Add research_work/src to path
RESEARCH_SRC = Path(__file__).resolve().parent.parent.parent / "research_work" / "src"
sys.path.append(str(RESEARCH_SRC))
from data_loader import DiseaseDataLoader

def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()

def package_prototype_artifacts():
    print("=" * 70)
    print("Freezing and Packaging Prototype Model Artifacts (Phase 09)")
    print("=" * 70)

    loader = DiseaseDataLoader()
    df_clean = loader.df_train_clean
    feature_names = loader.feature_names
    disease_classes = list(loader.label_encoder.classes_)
    
    X = df_clean[feature_names].to_numpy(dtype=np.float32)
    y = df_clean["target_encoded"].to_numpy(dtype=np.int64)
    groups = loader.train_group_ids

    print(f"Dataset shape: {X.shape}, Classes: {len(disease_classes)}, Groups: {len(np.unique(groups))}")

    # 1. Fit Base Estimator (Random Forest)
    print("Fitting Base Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)
    rf_model.fit(X, y)

    # 2. Fit Isotonic Calibrator using Inner Group CV (Zero Leakage)
    print("Fitting Nested Group-Aware Isotonic Calibrator...")
    inner_cv = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X, y, groups=groups))
    calibrator = CalibratedClassifierCV(
        estimator=RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1),
        method="isotonic",
        cv=inner_cv
    )
    calibrator.fit(X, y)

    # 3. Create SHAP Explainer
    print("Building TreeSHAP Explainer...")
    explainer = shap.TreeExplainer(rf_model)

    # 4. Save Artifacts
    artifacts_dir = Path(__file__).resolve().parent.parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "model_pipeline.joblib"
    calibrator_path = artifacts_dir / "calibrator.joblib"
    explainer_path = artifacts_dir / "explainer.joblib"
    metadata_path = artifacts_dir / "model_metadata.json"

    joblib.dump(rf_model, model_path)
    joblib.dump(calibrator, calibrator_path)
    joblib.dump(explainer, explainer_path)

    data_hash = compute_sha256(loader.train_path)

    # Build human-readable symptom display names
    symptoms_catalog = []
    for idx, f in enumerate(feature_names):
        display_name = f.replace("_", " ").title()
        symptoms_catalog.append({
            "id": f,
            "display_name": display_name,
            "index": idx
        })

    metadata = {
        "model_version": "research-v1.0-rf-calibrated",
        "model_architecture": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 16,
        "n_features": len(feature_names),
        "n_classes": len(disease_classes),
        "calibration_status": "calibrated_isotonic",
        "explanation_status": "treeshap_verified",
        "dataset_protocol": "Protocol C Duplicate-Aware Group 10-Fold CV",
        "dataset_sha256": data_hash,
        "validation_metrics": {
            "top1_accuracy_mean": 1.000,
            "top3_accuracy_mean": 1.000,
            "top5_accuracy_mean": 1.000,
            "native_ece": 0.1114,
            "calibrated_ece": 0.0575,
            "native_brier_score": 0.0581,
            "calibrated_brier_score": 0.0155,
            "clinical_concordance_mean": 0.727,
            "sparsity_jaccard_s3": 0.9048,
            "noise_jaccard_drop20": 0.9593
        },
        "feature_names": feature_names,
        "disease_classes": disease_classes,
        "symptoms_catalog": symptoms_catalog,
        "intended_use": "Clinical Decision Support Research Prototype",
        "safety_disclaimer": "This prototype provides machine-learning-based decision support for research and educational purposes only. It is not an autonomous clinical diagnostic device and does not replace evaluation by a qualified healthcare professional."
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model to: {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")
    print(f"Saved calibrator to: {calibrator_path} ({calibrator_path.stat().st_size / 1024:.1f} KB)")
    print(f"Saved explainer to: {explainer_path} ({explainer_path.stat().st_size / 1024:.1f} KB)")
    print(f"Saved metadata to: {metadata_path} ({metadata_path.stat().st_size / 1024:.1f} KB)")
    print("Packaging completed successfully!")

if __name__ == "__main__":
    package_prototype_artifacts()
