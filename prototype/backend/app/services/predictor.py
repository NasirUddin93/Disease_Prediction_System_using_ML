"""
Prediction Service — converts a symptom list into a ranked disease differential.
Uses the calibrated model from the registry.
"""
import logging
import time
from typing import Any

import numpy as np

from app.services.model_registry import registry
from app.schemas import DiseaseCandidate, PredictResponse

logger = logging.getLogger(__name__)

SAFETY_NOTICE = (
    "This application is an experimental machine-learning research prototype. "
    "It has not been validated for clinical diagnosis or emergency triage. "
    "Its outputs do not replace assessment by a qualified healthcare professional."
)


def predict(symptom_ids: list[str], top_k: int = 5) -> PredictResponse:
    """
    Build the canonical feature vector, run calibrated inference, and
    return the top-k disease candidates with probabilities.
    """
    if not registry.is_ready:
        raise RuntimeError("Model is not loaded. Cannot run inference.")

    # Build feature vector
    x, unknown = registry.build_feature_vector(symptom_ids)

    if unknown:
        logger.warning("Unknown symptom IDs ignored: %s", unknown)

    x_2d = x.reshape(1, -1)

    t0 = time.perf_counter()

    # Use calibrated model if available, else raw model
    if registry.calibrator_ready:
        proba = registry.calibrator.predict_proba(x_2d)[0]
        calibration_status = registry.metadata.get("calibration_status", "calibrated_isotonic")
    else:
        proba = registry.model.predict_proba(x_2d)[0]
        calibration_status = "uncalibrated_raw"

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info("Inference completed in %.2f ms", latency_ms)

    # Validate probability vector
    if not np.all(np.isfinite(proba)) or proba.min() < 0:
        raise ValueError("Invalid probability vector from model.")

    # Align probabilities with canonical disease class labels
    disease_proba = list(zip(registry.disease_classes, proba.tolist()))
    disease_proba.sort(key=lambda x: x[1], reverse=True)

    candidates: list[DiseaseCandidate] = []
    for rank, (disease, prob) in enumerate(disease_proba[:top_k], start=1):
        candidates.append(DiseaseCandidate(
            rank=rank,
            disease=disease.strip(),
            probability=round(float(prob), 4),
            calibration_status=calibration_status,
        ))

    return PredictResponse(
        model_version=registry.metadata.get("model_version", "research-v1.0"),
        active_symptoms_count=int(x.sum()),
        unknown_symptoms=unknown,
        calibration_status=calibration_status,
        candidates=candidates,
        safety_notice=SAFETY_NOTICE,
    )
