"""
Inference endpoints.
POST /api/v1/predict
POST /api/v1/explain
"""
import logging

from fastapi import APIRouter, HTTPException, status

from app.schemas import (
    ExplainRequest,
    ExplainResponse,
    PredictRequest,
    PredictResponse,
)
from app.services.model_registry import registry
from app.services import predictor, explainer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Inference"])


@router.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest) -> PredictResponse:
    """
    Validate a symptom list and return ranked disease candidates.

    - Unrecognised symptom IDs are reported in `unknown_symptoms` and ignored.
    - Probabilities are calibrated via Isotonic regression (Phase 07).
    - Safety notice is always included in the response.
    """
    if not registry.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts not loaded. Please try again later.",
        )

    # Validate that at least one known symptom is present
    known = [s for s in request.reported_symptoms if s in registry._feature_index]
    if not known:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "No recognised symptom IDs provided.",
                "unknown_symptoms": request.reported_symptoms,
                "hint": "Use GET /api/v1/symptoms to retrieve the valid catalogue.",
            },
        )

    try:
        result = predictor.predict(request.reported_symptoms, top_k=request.top_k)
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed. Please check the server logs.",
        ) from exc

    return result


@router.post("/explain", response_model=ExplainResponse)
def explain_endpoint(request: ExplainRequest) -> ExplainResponse:
    """
    Generate a TreeSHAP local explanation for a specific disease candidate.

    - The explainer uses the same frozen RF model used for prediction.
    - SHAP values explain model behaviour, NOT clinical causality.
    """
    if not registry.explainer_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SHAP explainer not loaded. Explanation unavailable.",
        )

    # Validate target disease
    clean_classes = [d.strip() for d in registry.disease_classes]
    if request.target_disease.strip() not in clean_classes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": f"Unknown disease class: '{request.target_disease}'",
                "hint": "Use GET /api/v1/diseases to retrieve supported disease labels.",
            },
        )

    try:
        result = explainer.explain(request.reported_symptoms, request.target_disease)
    except Exception as exc:
        logger.exception("Explanation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Explanation failed. Please check the server logs.",
        ) from exc

    return result
