"""
================================================================================
Data Loader & Leakage-Aware Splitting Pipeline
================================================================================
Implements end-to-end data ingestion, sanitization, validation, and three
evaluation splitting protocols:
- Protocol A: Original Kaggle split (Training.csv -> Testing.csv)
- Protocol B: Standard Stratified 10-Fold CV
- Protocol C: Duplicate-Aware Group 10-Fold CV (zero pattern leakage)
================================================================================
"""

import sys
from pathlib import Path
from typing import Tuple, Generator, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

sys.path.append(str(Path(__file__).resolve().parent))
from utilities import setup_logger, ensure_dir, save_json
from experiment_config import (
    RAW_TRAIN_PATH,
    RAW_TEST_PATH,
    PROCESSED_DATA_DIR,
    DEFAULT_CONFIG,
    Protocol,
)

logger = setup_logger("data_loader")


class DiseaseDataLoader:
    """Manages data loading, sanitization, and splitting under Protocols A, B, and C."""

    def __init__(self, train_path: Path = RAW_TRAIN_PATH, test_path: Path = RAW_TEST_PATH, config=DEFAULT_CONFIG):
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)
        self.config = config
        self.processed_dir = Path(PROCESSED_DATA_DIR)
        ensure_dir(self.processed_dir)

        self.df_train_clean: pd.DataFrame = None
        self.df_test_clean: pd.DataFrame = None
        self.feature_names: List[str] = []
        self.disease_names: List[str] = []
        self.label_encoder: LabelEncoder = LabelEncoder()
        self.train_group_ids: np.ndarray = None

        self._load_and_sanitize()

    def _load_and_sanitize(self) -> None:
        """Loads raw CSVs, prunes artifact columns, standardizes feature names, and validates."""
        logger.info(f"Ingesting raw data from {self.train_path} and {self.test_path}...")
        df_train = pd.read_csv(self.train_path)
        df_test = pd.read_csv(self.test_path)

        # 1. Prune artifact / unnamed columns
        df_train = df_train[[c for c in df_train.columns if not ("Unnamed" in c or c.strip() == "")]]
        df_test = df_test[[c for c in df_test.columns if not ("Unnamed" in c or c.strip() == "")]]

        # 2. Standardize column names (strip leading/trailing whitespace)
        df_train.columns = [c.strip() for c in df_train.columns]
        df_test.columns = [c.strip() for c in df_test.columns]

        # 3. Separate features and target
        target_col = self.config.target_column
        feature_cols = [c for c in df_train.columns if c != target_col]
        self.feature_names = feature_cols

        # 4. Strict data validation
        assert not df_train.isnull().values.any(), "Unexpected nulls in Training data after cleaning"
        assert not df_test.isnull().values.any(), "Unexpected nulls in Testing data after cleaning"
        assert set(df_train.columns) == set(df_test.columns), "Feature mismatch between Train and Test"

        # Verify all feature values are strictly binary 0 or 1
        train_vals = np.unique(df_train[feature_cols].to_numpy())
        assert set(train_vals).issubset({0, 1}), f"Non-binary feature values detected: {train_vals}"

        # 5. Fit LabelEncoder on combined classes
        all_diseases = sorted(df_train[target_col].unique())
        self.label_encoder.fit(all_diseases)
        self.disease_names = list(self.label_encoder.classes_)

        df_train = df_train.copy()
        df_test = df_test.copy()

        df_train["target_encoded"] = self.label_encoder.transform(df_train[target_col])
        df_test["target_encoded"] = self.label_encoder.transform(df_test[target_col])

        # 6. Compute Group IDs for unique symptom vectors in Training data
        group_tuples = pd.Series([tuple(x) for x in df_train[feature_cols].to_numpy()])
        group_ids, uniques = pd.factorize(group_tuples)
        self.train_group_ids = group_ids
        df_train["group_id"] = group_ids

        self.df_train_clean = df_train.copy()
        self.df_test_clean = df_test.copy()

        logger.info(f"Ingestion verified: {len(df_train)} train rows, {len(df_test)} test rows, "
                    f"{len(self.feature_names)} features, {len(self.disease_names)} classes, "
                    f"{len(uniques)} unique symptom pattern groups.")

        # Save clean processed files
        self._export_processed()

    def _export_processed(self) -> None:
        """Exports sanitized datasets and metadata to data/processed/."""
        self.df_train_clean.to_csv(self.processed_dir / "clean_training.csv", index=False)
        self.df_test_clean.to_csv(self.processed_dir / "clean_testing.csv", index=False)
        
        # Save unique patterns table
        unique_df = self.df_train_clean.drop_duplicates(subset=self.feature_names)
        unique_df.to_csv(self.processed_dir / "unique_symptom_patterns.csv", index=False)

        save_json(self.feature_names, self.processed_dir / "feature_names.json")
        save_json(self.disease_names, self.processed_dir / "disease_classes.json")
        logger.info(f"Exported clean datasets to {self.processed_dir}")

    # -------------------------------------------------------------------------
    # Protocol A: Original Kaggle Split
    # -------------------------------------------------------------------------
    def get_protocol_a_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Protocol A: Standard Kaggle split.
        Returns (X_train, y_train, X_test, y_test)
        """
        X_train = self.df_train_clean[self.feature_names].to_numpy(dtype=np.float32)
        y_train = self.df_train_clean["target_encoded"].to_numpy(dtype=np.int64)

        X_test = self.df_test_clean[self.feature_names].to_numpy(dtype=np.float32)
        y_test = self.df_test_clean["target_encoded"].to_numpy(dtype=np.int64)

        return X_train, y_train, X_test, y_test

    # -------------------------------------------------------------------------
    # Protocol B: Standard Stratified 10-Fold Cross Validation
    # -------------------------------------------------------------------------
    def get_protocol_b_folds(self) -> Generator[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Protocol B: Standard Stratified 10-Fold CV on Training.csv.
        Yields (fold_idx, X_train, y_train, X_val, y_val)
        """
        X = self.df_train_clean[self.feature_names].to_numpy(dtype=np.float32)
        y = self.df_train_clean["target_encoded"].to_numpy(dtype=np.int64)

        skf = StratifiedKFold(n_splits=self.config.n_splits, shuffle=True, random_state=self.config.random_seed)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            yield fold_idx, X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    # -------------------------------------------------------------------------
    # Protocol C: Duplicate-Aware Group 10-Fold Cross Validation
    # -------------------------------------------------------------------------
    def get_protocol_c_folds(self) -> Generator[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Protocol C: Duplicate-Aware Group 10-Fold CV.
        Guarantees that identical symptom vectors never cross train/val splits.
        Yields (fold_idx, X_train, y_train, X_val, y_val)
        """
        X = self.df_train_clean[self.feature_names].to_numpy(dtype=np.float32)
        y = self.df_train_clean["target_encoded"].to_numpy(dtype=np.int64)
        groups = self.train_group_ids

        sgkf = StratifiedGroupKFold(n_splits=self.config.n_splits, shuffle=True, random_state=self.config.random_seed)
        for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
            # Integrity check: verify zero group overlap
            train_g = set(groups[train_idx])
            val_g = set(groups[val_idx])
            assert len(train_g.intersection(val_g)) == 0, f"Leakage detected in Protocol C fold {fold_idx}!"
            yield fold_idx, X[train_idx], y[train_idx], X[val_idx], y[val_idx]
