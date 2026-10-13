"""
Explainer Service — TreeSHAP local explanations for individual predictions.
Explains the model's reasoning for a specific disease candidate.
"""
import logging
import time
from typing import Any

import numpy as np

from app.services.model_registry import registry
from app.schemas import ExplainResponse, SHAPFeature

logger = logging.getLogger(__name__)

SAFETY_NOTICE = (
    "This application is an experimental machine-learning research prototype. "
    "It has not been validated for clinical diagnosis or emergency triage. "
    "Its outputs do not replace assessment by a qualified healthcare professional."
)

EXPLANATION_NOTE = (
    "SHAP values explain how each symptom influenced this model's output for the "
    "selected disease — they reflect model behaviour, not clinical causality. "
    "A negative SHAP value means the symptom reduced the model's confidence in "
    "this disease; it does not confirm the symptom is clinically absent or protective."
)


def explain(symptom_ids: list[str], target_disease: str) -> ExplainResponse:
    """
    Compute TreeSHAP local explanation for the given symptom profile and target disease.
    """
    if not registry.explainer_ready:
        raise RuntimeError("SHAP explainer is not loaded.")

    if target_disease not in registry.disease_classes and \
       target_disease.strip() not in [d.strip() for d in registry.disease_classes]:
        raise ValueError(f"Unknown disease class: '{target_disease}'")

    # Find canonical class index
    disease_index: int | None = None
    for i, d in enumerate(registry.disease_classes):
        if d.strip() == target_disease.strip():
            disease_index = i
            break
    if disease_index is None:
        raise ValueError(f"Disease class not found: '{target_disease}'")

    # Build feature vector
    x, unknown = registry.build_feature_vector(symptom_ids)
    if unknown:
        logger.warning("Unknown symptoms in explain request: %s", unknown)

    x_2d = x.reshape(1, -1)

    t0 = time.perf_counter()

    # Compute SHAP values
    shap_values = registry.explainer.shap_values(x_2d)

    # shap_values is (n_classes, n_samples, n_features) for multi-class RF
    if isinstance(shap_values, list):
        sv_for_class = np.array(shap_values[disease_index])[0]  # shape: (132,)
    else:
        sv_for_class = shap_values[0, :, disease_index]  # shape: (132,)

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info("SHAP explanation completed in %.2f ms", latency_ms)

    # Get calibrated probability for target disease
    if registry.calibrator_ready:
        proba = registry.calibrator.predict_proba(x_2d)[0]
    else:
        proba = registry.model.predict_proba(x_2d)[0]
    target_prob = float(proba[disease_index])

    # Build feature lists sorted by |shap_value|
    features_with_shap = [
        (registry.feature_names[i], float(sv_for_class[i]), int(x[i]))
        for i in range(len(registry.feature_names))
    ]
    features_with_shap.sort(key=lambda t: abs(t[1]), reverse=True)

    # Top 10 supporting (positive shap) and top 10 opposing (negative shap)
    supporting: list[SHAPFeature] = []
    opposing: list[SHAPFeature] = []

    for sym_id, shap_val, present in features_with_shap:
        entry = SHAPFeature(
            symptom_id=sym_id,
            display_name=registry.get_display_name(sym_id),
            shap_value=round(shap_val, 5),
            present=bool(present),
            direction="supporting" if shap_val > 0 else "opposing",
        )
        if shap_val > 0 and len(supporting) < 10:
            supporting.append(entry)
        elif shap_val < 0 and len(opposing) < 10:
            opposing.append(entry)

        if len(supporting) >= 10 and len(opposing) >= 10:
            break

    return ExplainResponse(
        target_disease=target_disease.strip(),
        target_probability=round(target_prob, 4),
        model_version=registry.metadata.get("model_version", "research-v1.0"),
        explanation_version="treeshap-v1.0",
        supporting_symptoms=supporting,
        opposing_symptoms=opposing,
        explanation_note=EXPLANATION_NOTE,
        safety_notice=SAFETY_NOTICE,
    )
