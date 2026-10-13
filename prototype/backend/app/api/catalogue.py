"""
Health and catalogue endpoints.
GET /health
GET /api/v1/symptoms
GET /api/v1/diseases
GET /api/v1/model-info
"""
from fastapi import APIRouter

from app.services.model_registry import registry
from app.schemas import HealthResponse, ModelInfoResponse, SymptomInfo

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Verify the API and required model artifacts are available."""
    return HealthResponse(
        status="ok" if registry.is_ready else "degraded",
        model_loaded=registry.is_ready,
        calibrator_loaded=registry.calibrator_ready,
        explainer_loaded=registry.explainer_ready,
        artifact_version=registry.metadata.get("model_version", "unknown"),
        n_features=len(registry.feature_names),
        n_classes=len(registry.disease_classes),
    )


@router.get("/api/v1/symptoms", response_model=list[SymptomInfo], tags=["Catalogue"])
def list_symptoms() -> list[SymptomInfo]:
    """Return the canonical 132-symptom catalogue sorted alphabetically by display name."""
    catalog = registry.symptoms_catalog or [
        {"id": f, "display_name": f.replace("_", " ").title(), "index": i}
        for i, f in enumerate(registry.feature_names)
    ]
    return sorted(
        [SymptomInfo(**s) for s in catalog],
        key=lambda s: s.display_name,
    )


@router.get("/api/v1/diseases", response_model=list[str], tags=["Catalogue"])
def list_diseases() -> list[str]:
    """Return the 41 supported disease labels sorted alphabetically."""
    return sorted(d.strip() for d in registry.disease_classes)


@router.get("/api/v1/model-info", response_model=ModelInfoResponse, tags=["System"])
def model_info() -> ModelInfoResponse:
    """Return model architecture, validation metrics, and provenance."""
    m = registry.metadata
    return ModelInfoResponse(
        model_version=m.get("model_version", "unknown"),
        model_architecture=m.get("model_architecture", "unknown"),
        n_features=m.get("n_features", len(registry.feature_names)),
        n_classes=m.get("n_classes", len(registry.disease_classes)),
        calibration_status=m.get("calibration_status", "unknown"),
        explanation_status=m.get("explanation_status", "unknown"),
        dataset_protocol=m.get("dataset_protocol", "unknown"),
        validation_metrics=m.get("validation_metrics", {}),
        intended_use=m.get("intended_use", "Research use only"),
        safety_disclaimer=m.get(
            "safety_disclaimer",
            "Research prototype. Not for clinical use.",
        ),
    )
