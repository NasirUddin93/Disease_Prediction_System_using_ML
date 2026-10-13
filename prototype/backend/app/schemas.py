"""
Pydantic schemas for CDSS Prototype API (Phase 09).
All request and response models for symptom selection, prediction, and explanation.
"""
from typing import Any
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────── Request Models ────────────────────────────────

class PredictRequest(BaseModel):
    reported_symptoms: list[str] = Field(
        ...,
        min_length=1,
        max_length=132,
        description="List of canonical symptom IDs from the 132-feature catalogue.",
        examples=[["high_fever", "headache", "joint_pain"]],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of top candidate diseases to return (1–10).",
    )

    @field_validator("reported_symptoms")
    @classmethod
    def no_duplicates(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for s in v:
            s = s.strip()
            if s and s not in seen:
                seen.add(s)
                unique.append(s)
        return unique


class ExplainRequest(BaseModel):
    reported_symptoms: list[str] = Field(
        ...,
        min_length=1,
        max_length=132,
        description="Same symptom list used for the prediction to explain.",
    )
    target_disease: str = Field(
        ...,
        description="The disease class to generate the explanation for.",
    )

    @field_validator("reported_symptoms")
    @classmethod
    def no_duplicates(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for s in v:
            s = s.strip()
            if s and s not in seen:
                seen.add(s)
                unique.append(s)
        return unique


# ─────────────────────────── Response Models ───────────────────────────────

class SymptomInfo(BaseModel):
    id: str
    display_name: str
    index: int


class DiseaseCandidate(BaseModel):
    rank: int
    disease: str
    probability: float
    calibration_status: str


class PredictResponse(BaseModel):
    model_version: str
    active_symptoms_count: int
    unknown_symptoms: list[str]
    calibration_status: str
    candidates: list[DiseaseCandidate]
    safety_notice: str


class SHAPFeature(BaseModel):
    symptom_id: str
    display_name: str
    shap_value: float
    present: bool
    direction: str  # "supporting" | "opposing"


class ExplainResponse(BaseModel):
    target_disease: str
    target_probability: float
    model_version: str
    explanation_version: str
    supporting_symptoms: list[SHAPFeature]
    opposing_symptoms: list[SHAPFeature]
    explanation_note: str
    safety_notice: str


class ModelInfoResponse(BaseModel):
    model_version: str
    model_architecture: str
    n_features: int
    n_classes: int
    calibration_status: str
    explanation_status: str
    dataset_protocol: str
    validation_metrics: dict[str, Any]
    intended_use: str
    safety_disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    calibrator_loaded: bool
    explainer_loaded: bool
    artifact_version: str
    n_features: int
    n_classes: int
