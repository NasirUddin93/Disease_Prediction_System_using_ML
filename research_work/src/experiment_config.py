"""
================================================================================
Experiment Configuration: Paths, Protocols, Model Registry & Constants
================================================================================
Centralized configuration for leakage-aware disease prediction experiments.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

# Root project paths
SRC_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SRC_DIR.parent
PROJECT_ROOT = RESEARCH_DIR.parent

RAW_TRAIN_PATH = RESEARCH_DIR / "data" / "raw" / "Training.csv"
RAW_TEST_PATH = RESEARCH_DIR / "data" / "raw" / "Testing.csv"
PROCESSED_DATA_DIR = RESEARCH_DIR / "data" / "processed"

REPORTS_DIR = RESEARCH_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CONFUSION_DIR = REPORTS_DIR / "confusion_matrices"

PAPER_FIGURES_DIR = PROJECT_ROOT / "Paper_Writing" / "figures"

# Core Experiment Protocols
class Protocol:
    A_KAGGLE = "Protocol_A_Kaggle_Split"
    B_STRATIFIED = "Protocol_B_Stratified_10Fold"
    C_GROUP = "Protocol_C_DuplicateAware_10Fold"

ALL_PROTOCOLS = [
    Protocol.A_KAGGLE,
    Protocol.B_STRATIFIED,
    Protocol.C_GROUP,
]

@dataclass
class ExperimentConfig:
    random_seed: int = 42
    n_splits: int = 10
    target_column: str = "prognosis"
    artifact_column: str = "Unnamed: 133"
    
    # Metrics to track for all models and protocols
    tracked_metrics: List[str] = field(default_factory=lambda: [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "top_3_accuracy",
        "top_5_accuracy",
        "train_time_sec",
        "inference_time_ms_per_sample",
    ])

DEFAULT_CONFIG = ExperimentConfig()
