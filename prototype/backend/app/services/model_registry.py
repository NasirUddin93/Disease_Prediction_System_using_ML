"""
Model Registry — loads and exposes all frozen artifacts as singletons.
Called once at application startup; all routers share these references.
"""
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Singleton registry for all frozen Phase 09 artifacts."""

    def __init__(self) -> None:
        self.model: Any = None
        self.calibrator: Any = None
        self.explainer: Any = None
        self.metadata: dict = {}
        self.feature_names: list[str] = []
        self.disease_classes: list[str] = []
        self.symptoms_catalog: list[dict] = []
        self._feature_index: dict[str, int] = {}

    def load(self, artifacts_dir: Path) -> None:
        """Load all artifacts from the given directory. Raises on failure."""
        logger.info("Loading model artifacts from %s …", artifacts_dir)

        # Metadata (must load first — provides feature_names/disease_classes)
        meta_path = artifacts_dir / "model_metadata.json"
        with open(meta_path) as fh:
            self.metadata = json.load(fh)

        self.feature_names = self.metadata["feature_names"]
        self.disease_classes = self.metadata["disease_classes"]
        self.symptoms_catalog = self.metadata.get("symptoms_catalog", [])
        self._feature_index = {f: i for i, f in enumerate(self.feature_names)}

        # Model pipeline
        model_path = artifacts_dir / "model_pipeline.joblib"
        self.model = joblib.load(model_path)
        logger.info("Model loaded: %s (%d features, %d classes)",
                    self.metadata.get("model_architecture", "unknown"),
                    len(self.feature_names), len(self.disease_classes))

        # Calibrator
        calibrator_path = artifacts_dir / "calibrator.joblib"
        self.calibrator = joblib.load(calibrator_path)
        logger.info("Calibrator loaded: %s", self.metadata.get("calibration_status"))

        # SHAP explainer
        explainer_path = artifacts_dir / "explainer.joblib"
        self.explainer = joblib.load(explainer_path)
        logger.info("SHAP explainer loaded: %s", self.metadata.get("explanation_status"))

        logger.info("All artifacts loaded successfully.")

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    @property
    def calibrator_ready(self) -> bool:
        return self.calibrator is not None

    @property
    def explainer_ready(self) -> bool:
        return self.explainer is not None

    def build_feature_vector(self, symptom_ids: list[str]) -> tuple[np.ndarray, list[str]]:
        """
        Convert a list of reported symptom IDs into the canonical 132-dim binary vector.
        Returns (vector, unknown_symptom_ids).
        Unknown symptoms are ignored with a warning.
        """
        x = np.zeros(len(self.feature_names), dtype=np.float32)
        unknown: list[str] = []
        for sid in symptom_ids:
            idx = self._feature_index.get(sid)
            if idx is not None:
                x[idx] = 1.0
            else:
                unknown.append(sid)
        return x, unknown

    def get_display_name(self, symptom_id: str) -> str:
        for s in self.symptoms_catalog:
            if s["id"] == symptom_id:
                return s["display_name"]
        return symptom_id.replace("_", " ").title()


# Module-level singleton
registry = ModelRegistry()
