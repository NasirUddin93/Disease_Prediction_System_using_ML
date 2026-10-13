# Phase 09 Completion Plan: Clinical Decision Support System (CDSS) Prototype (Experiment E9)

**Document Status**: Official Engineering & Integration Specification  
**Project**: *Disease Prediction System Using Machine Learning: A Robust, Explainable, and Uncertainty-Aware Framework Under Real-World Symptom Sparsity and Noise*  
**Candidate**: Nasir uddin (Registration No: `23880300014`, Session: `2023-2024`)  
**Supervisor**: Husne Ara Rubaiyeat, Assistant Professor, Department of Computer Science, National University, Bangladesh  
**Target Chapter**: `Book_Writing/Chap7/Chapter7.tex` (Section 7.4 Deployment Roadmap & Prototype Evaluation)  

---

## 1. Executive Objective

The objective of Phase 09 (Experiment E9) is to integrate the empirical machine-learning research outputs from Phases 00–08 into a transparent, interactive research prototype. The system accepts patient-reported symptoms, generates ranked differential candidate diseases (Top-3 / Top-5), displays verified post-hoc calibrated probabilities (from Phase 07), and provides attributable SHAP evidence (from Phase 08) accompanied by explicit clinical-safety guardrails.

> [!IMPORTANT]
> The prototype connects validated research artifacts; it does **not** introduce an ad-hoc unvalidated model. Calibrated probabilities and SHAP explanations will only be surfaced once Phases 07 and 08 have completed full Protocol C validation.

---

## 2. System Architecture & Component Design

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Frontend Interface (React / Next.js)                 │
│  - Symptom Multi-Select (Canonical 132-Feature Schema)                 │
│  - Differential Diagnosis Ranking (Top-3 / Top-5 Candidates)           │
│  - Calibrated Probability Gauges & Uncertainty Breakdown               │
│  - Local SHAP Evidence Display (Supporting vs. Missing Symptoms)       │
│  - Prominent Medical Decision-Support Safety Notices                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ REST API (JSON)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend Service                         │
│  - Endpoint Validation & Pydantic Contracts (/predict, /explain)       │
│  - Strict Feature Alignment (Zero-Index Imputation for Unreported)     │
│  - Model Management & Metadata Provenance Service                      │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
                    ▼                                ▼
┌─────────────────────────────────────┐  ┌───────────────────────────────┐
│         Inference Engine            │  │      SHAP Explainer Module    │
│  - Frozen Estimator (Protocol C)    │  │  - TreeSHAP Background Cache  │
│  - Fold-Isolated k=50 Selector      │  │  - Patient Waterfall Breakdown│
│  - Post-Hoc Calibrator (Phase 07)   │  │  - Guideline Concordance Flags│
└─────────────────────────────────────┘  └───────────────────────────────┘
```

---

## 3. Ten-Step Implementation Workflow

### Step 1. Define Scope & Data Integrity Guardrails
* **Feature Schema**: Strictly enforces the canonical 132 binary symptom catalog from `research_work/data/processed`.
* **Binary Representation Principle**: The UI and API explicitly distinguish between *reported positive* ($1$) and *unreported/unselected* ($0$) symptoms, warning users that $0$ does not clinically prove symptom absence.
* **Class Nomenclature**: Uses canonical 41 disease class labels directly from the trained label encoder.

### Step 2. Project Directory Structure
```
prototype/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application entrypoint & CORS
│   │   ├── schemas.py           # Pydantic request/response schemas
│   │   ├── api/
│   │   │   ├── health.py        # /health, /api/v1/model-info
│   │   │   ├── symptoms.py      # /api/v1/symptoms, /api/v1/diseases
│   │   │   └── predictions.py   # /api/v1/predict, /api/v1/explain
│   │   ├── services/
│   │   │   ├── predictor.py     # Inference pipeline execution
│   │   │   ├── calibrator.py    # Multi-class calibrated probability scaling
│   │   │   └── explainer.py     # SHAP local feature attribution service
│   │   └── core/
│   │       ├── config.py        # API settings & paths
│   │       └── model_loader.py  # Frozen artifact deserializer
│   ├── tests/                   # Pytest API contracts & inference test cases
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/                     # React/Next.js components
│   │   ├── components/          # SymptomSelector, CandidateCard, SHAPWaterfall, SafetyBanner
│   │   └── services/api.ts      # Typed REST client
│   └── package.json
├── artifacts/                   # Serialized research models & metadata
│   ├── model_pipeline.joblib
│   ├── calibrator.joblib
│   ├── explainer_cache.joblib
│   └── model_metadata.json
└── README.md
```

### Step 3. Model Selection & Artifact Freezing
* Model selection criteria based on comprehensive empirical findings across Phases 00–08:
  1. High Protocol C generalization (minimizing memorization).
  2. Graceful degradation under symptom sparsity ($s \le 5$) and noise ($10\% \to 20\%$).
  3. Verified Expected Calibration Error (ECE) reduction via Platt/Isotonic scaling.
  4. High cross-fold explanation stability (Jaccard $\ge 0.70$).
* Serialized bundle includes: estimator, feature selector, label encoder, calibrator, and `model_metadata.json` (SHA-256 data hash, validation metrics, training date).

### Step 4. REST API Contract Specification
* **`GET /health`**: Verifies backend availability and loaded model artifact integrity.
* **`GET /api/v1/symptoms`**: Returns sorted 132 symptom identifiers and human-readable display labels.
* **`GET /api/v1/diseases`**: Returns supported 41 disease classes.
* **`GET /api/v1/model-info`**: Returns model architecture, training protocol, calibration mode, and scientific limitations.
* **`POST /api/v1/predict`**:
  * Request:
    ```json
    {
      "reported_symptoms": ["high_fever", "headache", "joint_pain"],
      "top_k": 5
    }
    ```
  * Response:
    ```json
    {
      "model_version": "research-v1.0-rf-calibrated",
      "active_symptoms_count": 3,
      "calibration_status": "calibrated_sigmoid",
      "candidates": [
        {"rank": 1, "disease": "Dengue", "probability": 0.742, "log_odds": 1.05},
        {"rank": 2, "disease": "Malaria", "probability": 0.148, "log_odds": -1.75},
        {"rank": 3, "disease": "Typhoid", "probability": 0.052, "log_odds": -2.90}
      ],
      "safety_notice": "Research decision support only; not an autonomous clinical diagnosis."
    }
    ```
* **`POST /api/v1/explain`**:
  * Deconstructs top predicted candidates into positive supporting symptoms ($\phi_j > 0$) and opposing/absent features ($\phi_j < 0$).

### Step 5. Frontend User Experience Design
* **Screen 1 (Symptom Search & Intake)**: Fast searchable multi-select with instant chip removal, symptom count badge, and "Reported vs. Confirmed Negative" guidance.
* **Screen 2 (Differential Candidate Ranking)**: Top-1 primary candidate card, followed by ordered Top-3 and Top-5 differential candidates with calibrated percentage badges.
* **Screen 3 (Explainability Evidence View)**: Interactive horizontal bar / waterfall visual displaying exact feature attributions ($\phi_j$) with clear medical interpretability guidance.
* **Persistent Safety Banner**: Prominently displayed across all screens:
  > *"This prototype provides machine-learning-based decision support for research and educational purposes only. It is not an autonomous diagnostic device and does not replace evaluation by a qualified healthcare professional."*

### Step 6. Medical Explanation Boundaries
* UI uses rigorous attribution phrasing: *"These reported symptoms increased model likelihood for this disease"* rather than prescriptive claims: *"These symptoms prove disease X"*.
* Notes that SHAP measures feature attribution within the benchmark model, not pathophysiological causation.

### Step 7. Automated Verification & Testing
* Unit tests for symptom vectorization (132-dimensional binary encoding).
* Integration tests verifying probability sums ($\sum P_c \approx 1.0$) and candidate ranking alignment.
* Out-of-distribution input tests (empty symptoms, unknown strings, all symptoms positive).

### Step 8. Latency & Resource Benchmarking
* Evaluate inference latency (p50, p95), SHAP attribution computation latency, and memory footprint under concurrent load.
* Document findings in `research_work/reports/phase9_prototype_report.md`.

### Step 9. Reproducibility & Containerization
* Provide `Dockerfile` and `docker-compose.yml` for single-command deployment.
* Ensure all model artifacts are versioned and tied to git commit hashes.

### Step 10. Thesis & Repository Synthesis
* Update Chapter VII of the thesis manuscript with prototype architecture diagrams, latency metrics, and usability findings.
* Update `Overview/research_consistency_overview.md` to officially mark Phase 09 complete upon passing all acceptance criteria.

---

## 4. Acceptance Checklist

- [ ] Frozen model artifacts generated from validated Protocol C pipeline.
- [ ] Calibrated probabilities from Phase 07 strictly integrated.
- [ ] SHAP explanations from Phase 08 connected with attribution phrasing.
- [ ] FastAPI backend routes (`/health`, `/symptoms`, `/predict`, `/explain`) tested and passing.
- [ ] Responsive frontend with symptom search, candidate ranking, and waterfall displays.
- [ ] Comprehensive safety and non-diagnostic disclaimers implemented.
- [ ] `research_work/reports/phase9_prototype_report.md` written and verified.
- [ ] Manuscript Chapter VII updated with prototype findings.
