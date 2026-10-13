"""
Phase 09 — Backend API Test Suite.
Tests: health, catalogue endpoints, predict, explain, and error cases.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app

# Use the TestClient as a context manager to properly trigger the lifespan
# (and thus the model loading). conftest.py pre-loads artifacts as well.
client = TestClient(app, raise_server_exceptions=False)


# ─────────────────────────── Health ────────────────────────────────────────

def test_health_returns_200():
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_payload_has_required_fields():
    data = client.get("/health").json()
    for field in ("status", "model_loaded", "calibrator_loaded", "explainer_loaded",
                  "n_features", "n_classes"):
        assert field in data, f"Missing field: {field}"


def test_health_model_loaded():
    data = client.get("/health").json()
    assert data["model_loaded"] is True, "Model should be loaded"
    assert data["n_features"] == 132
    assert data["n_classes"] == 41


# ─────────────────────────── Catalogue ─────────────────────────────────────

def test_symptoms_returns_list():
    resp = client.get("/api/v1/symptoms")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 132


def test_symptoms_have_canonical_fields():
    data = client.get("/api/v1/symptoms").json()
    for item in data[:5]:
        assert "id" in item
        assert "display_name" in item
        assert "index" in item


def test_diseases_returns_41_classes():
    resp = client.get("/api/v1/diseases")
    assert resp.status_code == 200
    assert len(resp.json()) == 41


def test_model_info_fields():
    resp = client.get("/api/v1/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_features"] == 132
    assert data["n_classes"] == 41
    assert "validation_metrics" in data


# ─────────────────────────── Predict ───────────────────────────────────────

DENGUE_SYMPTOMS = [
    "high_fever", "headache", "joint_pain", "pain_behind_the_eyes",
    "nausea", "fatigue", "skin_rash",
]


def test_predict_returns_200():
    resp = client.post("/api/v1/predict", json={"reported_symptoms": DENGUE_SYMPTOMS})
    assert resp.status_code == 200


def test_predict_default_top_k_is_5():
    data = client.post("/api/v1/predict", json={"reported_symptoms": DENGUE_SYMPTOMS}).json()
    assert len(data["candidates"]) == 5


def test_predict_custom_top_k():
    data = client.post(
        "/api/v1/predict",
        json={"reported_symptoms": DENGUE_SYMPTOMS, "top_k": 3},
    ).json()
    assert len(data["candidates"]) == 3


def test_predict_ranks_are_sequential():
    data = client.post("/api/v1/predict", json={"reported_symptoms": DENGUE_SYMPTOMS}).json()
    ranks = [c["rank"] for c in data["candidates"]]
    assert ranks == list(range(1, len(ranks) + 1))


def test_predict_probabilities_are_valid():
    data = client.post("/api/v1/predict", json={"reported_symptoms": DENGUE_SYMPTOMS}).json()
    for c in data["candidates"]:
        assert 0.0 <= c["probability"] <= 1.0


def test_predict_has_safety_notice():
    data = client.post("/api/v1/predict", json={"reported_symptoms": DENGUE_SYMPTOMS}).json()
    assert "safety_notice" in data
    assert len(data["safety_notice"]) > 20


def test_predict_reports_unknown_symptoms():
    resp = client.post(
        "/api/v1/predict",
        json={"reported_symptoms": ["high_fever", "UNKNOWN_SYMPTOM_XYZ"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "UNKNOWN_SYMPTOM_XYZ" in data["unknown_symptoms"]


def test_predict_all_unknown_returns_422():
    resp = client.post(
        "/api/v1/predict",
        json={"reported_symptoms": ["not_a_symptom", "also_invalid"]},
    )
    assert resp.status_code == 422


def test_predict_empty_list_returns_422():
    resp = client.post("/api/v1/predict", json={"reported_symptoms": []})
    assert resp.status_code == 422


def test_predict_top_k_zero_returns_422():
    resp = client.post(
        "/api/v1/predict",
        json={"reported_symptoms": DENGUE_SYMPTOMS, "top_k": 0},
    )
    assert resp.status_code == 422


def test_predict_deduplicates_symptoms():
    data = client.post(
        "/api/v1/predict",
        json={"reported_symptoms": ["high_fever", "high_fever", "headache"]},
    ).json()
    assert data["active_symptoms_count"] == 2


# ─────────────────────────── Explain ───────────────────────────────────────

def test_explain_returns_200():
    resp = client.post(
        "/api/v1/explain",
        json={"reported_symptoms": DENGUE_SYMPTOMS, "target_disease": "Dengue"},
    )
    assert resp.status_code == 200


def test_explain_has_required_fields():
    data = client.post(
        "/api/v1/explain",
        json={"reported_symptoms": DENGUE_SYMPTOMS, "target_disease": "Dengue"},
    ).json()
    for field in ("target_disease", "target_probability", "supporting_symptoms",
                  "opposing_symptoms", "explanation_note", "safety_notice"):
        assert field in data, f"Missing field: {field}"


def test_explain_shap_features_have_direction():
    data = client.post(
        "/api/v1/explain",
        json={"reported_symptoms": DENGUE_SYMPTOMS, "target_disease": "Dengue"},
    ).json()
    for feat in data["supporting_symptoms"]:
        assert feat["direction"] == "supporting"
        assert feat["shap_value"] > 0
    for feat in data["opposing_symptoms"]:
        assert feat["direction"] == "opposing"
        assert feat["shap_value"] < 0


def test_explain_unknown_disease_returns_422():
    resp = client.post(
        "/api/v1/explain",
        json={"reported_symptoms": DENGUE_SYMPTOMS, "target_disease": "NotARealDisease"},
    )
    assert resp.status_code == 422
