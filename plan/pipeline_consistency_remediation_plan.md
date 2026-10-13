# Pipeline Consistency Remediation Plan
## Detailed Strategy to Remove All Inconsistencies Across Plan $\to$ Research Work $\to$ Prototype $\to$ Book Writing

**Project Title**: *Disease Prediction System Using Machine Learning: A Robust, Explainable, and Uncertainty-Aware Framework Under Real-World Symptom Sparsity and Noise*  
**Degree**: Master in Advanced Studies (MAS) in Computer Science  
**Department**: Department of Computer Science, National University, Gazipur, Bangladesh  
**Candidate**: Nasir uddin (Registration No: `23880300014`, Session: `2023-2024`)  
**Supervisor**: Husne Ara Rubaiyeat, Assistant Professor  
**Document Creation Date**: October 2026  

---

## 1. Executive Summary & Problem Formulation

A comprehensive audit of the research suite revealed that while the **Plan**, **Empirical Research Work**, and **Phase 09 Prototype** are 100% complete and fully verified, the **Book Writing (LaTeX thesis)** contains critical synchronization lags:

1. **Chapter VII Relegates Finished Research to "Future Work"**: Probability Calibration (Phase 07 / Hypothesis H4) and TreeSHAP Clinical Literature Concordance (Phase 08 / Hypothesis H5) are currently listed under *Future Work and Research Roadmap* ([Chapter7.tex:L124–L131](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/Book_Writing/Chap7/Chapter7.tex#L124-L131)), even though both empirical experiments are finished and their figures already exist in `Book_Writing/figures/`.
2. **Phase 09 CDSS Prototype is Omitted from Thesis**: The functional FastAPI backend (port 8181), glassmorphism UI, and 22 passing pytest unit tests are not yet documented in the thesis body.
3. **Abstract Contradicts Chapter VII**: [Abstract.tex](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/Book_Writing/Abstract/Abstract.tex) claims reporting noise is a "future planned methodology", while Chapter VII already contains the full 20,520-run noise benchmark.
4. **Overview Audit Docs are Outdated**: Older audit files list Phase 09 as "Blueprint Ready" or unexecuted.

This document details the exact, step-by-step remediation procedure to resolve every inconsistency.

---

## 2. Structural Architecture Decision

In the thesis manuscript, two structural approaches are possible:

### Option A: 7-Chapter Consolidated Structure (Recommended)
* Retain the existing 7 chapters approved under the National University MAS format (`rac.sty`).
* Upgrade **Chapter VII** from a basic discussion into a comprehensive synthesis chapter:
  $$\textbf{Chapter VII: Advanced Clinical Robustness, Explainability, Prototype, and Discussion}$$
* Sections within Chapter VII:
  1. `\section{Discussion and Scientific Limitations}`
  2. `\section{Stochastic Reporting Noise and Error Asymmetry (Hypothesis H3)}` *(Existing)*
  3. `\section{Probability Calibration and Uncertainty Quantification (Hypothesis H4)}` *(New - Promoted from Future Work)*
  4. `\section{Explainable AI and Clinical Literature Concordance (Hypothesis H5)}` *(New - Promoted from Future Work)*
  5. `\section{Clinical Decision Support System Research Prototype (Phase 09)}` *(New)*
  6. `\section{Comprehensive Conclusions and Key Findings Synthesis}` *(Expanded to cover H1–H5 and Prototype)*
  7. `\section{Future Work and Research Roadmap}` *(Restricted to true clinical extensions)*
* **Advantages**: Zero risk of breaking chapter numbering, equation numbering, table of contents, or `main.tex` cross-references.

### Option B: 8-Chapter Split (Matching `plan/research_plan.md`)
* Split Chapter VII into two chapters: Chapter VII (CDSS Prototype) and Chapter VIII (Discussion & Conclusion).
* **Tradeoff**: Requires modifying `main.tex`, updating TOC anchors, and re-indexing references.

> **Selected Path**: **Option A** is the recommended operational path. It provides 100% scientific completeness with zero structural disruption.

---

## 3. Step-by-Step Remediation Action Plan

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              REMEDIATION EXECUTION PHASES                              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Step 1: Promote Phase 07 (Probability Calibration & H4) into Chapter VII               │
│ Step 2: Promote Phase 08 (TreeSHAP Concordance & H5) into Chapter VII                  │
│ Step 3: Add Phase 09 (CDSS Prototype Architecture & Evaluation) to Chapter VII         │
│ Step 4: Expand Chapter VII Conclusion across all 5 Hypotheses & Prototype              │
│ Step 5: Refine Chapter VII Future Work to Authentic Clinical Trials                    │
│ Step 6: Harmonize Book_Writing/Abstract/Abstract.tex                                   │
│ Step 7: Update and Re-package the Overleaf LaTeX Archive (.zip)                        │
│ Step 8: Update Repository Overview Documentation                                       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Step 1: Integrate Phase 07 (Probability Calibration — Hypothesis H4) into Chapter VII

* **File**: `Book_Writing/Chap7/Chapter7.tex`
* **Source Data**: `research_work/reports/phase7_calibration_report.md` ($N = 2,700$ fold runs across Protocol C Duplicate-Aware Group 10-Fold CV).
* **Figures to Embed**:
  - `\includegraphics[width=\textwidth]{reliability_diagrams.png}`
  - `\includegraphics[width=0.85\textwidth]{calibration_metric_comparison.png}`
* **Text & Data Specifications**:
  1. **Background**: Medical triage requires that confidence probabilities reflect empirical likelihoods ($P(\text{Dengue}) = 0.90 \implies 90\%$ prevalence).
  2. **Quantitative Table**:
     - **Random Forest**: Native ECE $= 0.1114 \pm 0.0218 \longrightarrow$ Isotonic ECE $= \mathbf{0.0575 \pm 0.0278}$ (Brier Score drops $0.0581 \to 0.0155$).
     - **MLP Neural Net**: Native ECE $= 0.2680 \pm 0.0752 \longrightarrow$ Isotonic ECE $= \mathbf{0.0517 \pm 0.0275}$ (Brier Score drops $0.1294 \to 0.0157$).
     - **Extra Trees**: Native ECE $= 0.1042 \pm 0.0327 \longrightarrow$ Isotonic ECE $= \mathbf{0.0629 \pm 0.0283}$ (Brier Score drops $0.0838 \to 0.0186$).
     - **Decision Tree**: Native ECE $= 0.1743 \longrightarrow$ Isotonic ECE $= 0.3304$ (overfitting in calibration fold).
  3. **Hypothesis H4 Verdict**: **Partially Supported (Architecture-Dependent)** — post-hoc non-parametric isotonic regression significantly minimizes ECE and Brier score across tree ensembles and neural architectures without degrading classification accuracy, but fails on unregularized single decision trees.

---

### Step 2: Integrate Phase 08 (Explainable AI & Clinical Concordance — Hypothesis H5) into Chapter VII

* **File**: `Book_Writing/Chap7/Chapter7.tex`
* **Source Data**: `research_work/reports/phase8_shap_report.md` (TreeSHAP and LinearSHAP across Protocol C folds).
* **Figures to Embed**:
  - `\includegraphics[width=0.85\textwidth]{shap_clinical_concordance.png}`
  - `\includegraphics[width=0.85\textwidth]{shap_stability_comparison.png}`
* **Text & Data Specifications**:
  1. **Cross-Fold Attribution Stability**:
     - Ensemble architectures demonstrate strong explanation stability across folds: Random Forest Top-5 Jaccard $= 0.5265 \pm 0.2500$; XGBoost $= 0.6435 \pm 0.2984$.
     - Single Decision Tree collapses: Jaccard $= 0.1030 \pm 0.1273$ (Spearman rank correlation $-0.6033$).
     - Perturbation stability under noise ($0.9593$ Jaccard) and symptom sparsity ($0.9048$ Jaccard).
  2. **Authoritative Medical Literature Concordance Audit**:
     - Systematically cross-reference top 5 SHAP symptoms against diagnostic guidelines from **WHO**, **CDC**, **ADA**, **ACG**, and **GINA**:
       - **Malaria (CDC)**: $100\%$ concordance (*muscle pain, diarrhoea, sweating, chills, high fever*).
       - **Typhoid (WHO)**: $100\%$ concordance (*toxic look, belly pain, constipation, diarrhoea, chills*).
       - **GERD (ACG)**: $100\%$ concordance (*ulcers on tongue, stomach pain, acidity, chest pain, cough*).
       - **Diabetes (ADA)**: $80\%$ concordance (*polyuria, increased appetite, irregular sugar level, weight loss*).
       - **Dengue (WHO)**: $60\%$ concordance (*pain behind eyes, back pain, muscle pain*).
       - **Bronchial Asthma (GINA)**: $60\%$ concordance (*mucoid sputum, breathlessness, cough*).
       - **Cohort Mean Concordance**: **72.7%**.
  3. **Hypothesis H5 Verdict**: **Fully Supported** — model decisions align with recognized clinical criteria, and ensemble architectures exhibit significantly higher cross-fold attribution stability than single-tree baselines.

---

### Step 3: Add Phase 09 (CDSS Prototype Architecture & Evaluation) to Chapter VII

* **File**: `Book_Writing/Chap7/Chapter7.tex`
* **Source Data**: `prototype/` codebase, API specifications, and automated test suite.
* **Text Specifications**:
  1. **Architecture & Service Layer**:
     - FastAPI asynchronous microservice running on port `8181`.
     - Serialized research pipelines: `model_pipeline.joblib` (Random Forest), `calibrator.joblib` (Isotonic CalibratedClassifierCV), and `explainer.joblib` (TreeExplainer).
  2. **Responsible Clinical Presentation Paradigm**:
     - Avoidance of deterministic single-label statements.
     - Multi-candidate differential ranking: Top-1, Top-3, Top-5 candidates with calibrated probabilities.
     - Directional feature evidence: positive supporting evidence ($+$ SHAP) vs. absent/opposing evidence ($-$ SHAP).
     - Prominent non-diagnostic research warning and triage advisory notice.
  3. **Verification & Performance Metrics**:
     - Sub-15ms response latency for full prediction + SHAP explanation cycle.
     - 22 passing automated test cases (`tests/test_api.py`) verifying schema validation, probability calibration bounds, top-$k$ integrity, and error boundaries.

---

### Step 4: Expand Chapter VII Conclusion Across all 5 Hypotheses & Prototype

* **File**: `Book_Writing/Chap7/Chapter7.tex`
* **Action**: Update `\section{Conclusion}` and `\section{Key Findings Summary}` to present the unified 7-point scientific synthesis:
  1. **Phase 0 Forensics**: 93.8% synthetic duplication and 97.6% train-test leakage exposed.
  2. **Phase 2–3 Protocols**: Protocol C exposes tree model collapse (Decision Tree: $-17.44\%$, XGBoost: $-12.80\%$), confirming Hypothesis H1.
  3. **Phase 4 Feature Selection**: 50 consensus symptoms retain $\ge 95\%$ Macro-F1.
  4. **Phase 5 Sparsity**: Top-3/Top-5 candidate ranking retains $\ge 88.3\%$ and $\ge 92.7\%$ inclusion at $s=3$, confirming Hypothesis H2.
  5. **Phase 6 Reporting Noise**: Noise asymmetry is model-dependent under matched error burden, partially supporting Hypothesis H3.
  6. **Phase 7 Probability Calibration**: Isotonic regression cuts Random Forest ECE from $0.1114 \to 0.0575$ and MLP ECE from $0.2680 \to 0.0517$, partially supporting Hypothesis H4.
  7. **Phase 8 & 9 XAI & Prototype**: 72.7% clinical guideline concordance confirms Hypothesis H5, successfully realized in an operational CDSS research prototype.

---

### Step 5: Restrict "Future Work" to True Prospective Clinical Extensions

* **File**: `Book_Writing/Chap7/Chapter7.tex`
* **Action**: Cleanly separate completed algorithmic research from prospective medical extensions:
  1. **Prospective Multi-Center Cohort Validation**: Moving beyond rule-derived synthetic benchmarks to validate on electronic health records (EHR) with authentic demographic diversity.
  2. **Multi-Modal Diagnostic Integration**: Combining patient-reported subjective symptoms with objective continuous vitals, laboratory blood telemetry, and imaging biomarkers.
  3. **Dynamic Longitudinal Trajectory Tracking**: Replacing static binary symptom vectors with temporal onset tracking and symptom severity grading.

---

### Step 6: Harmonize `Book_Writing/Abstract/Abstract.tex`

* **File**: `Book_Writing/Abstract/Abstract.tex`
* **Action**: Replace the obsolete future roadmap sentence with the completed empirical findings:

```latex
% Replace lines concluding the Abstract with:
Furthermore, the framework rigorously evaluates bidirectional stochastic reporting noise (demonstrating model-dependent error asymmetry under matched error burdens), empirical probability calibration (reducing Expected Calibration Error by up to 80\% via non-parametric isotonic regression), and Explainable AI (achieving 72.7\% clinical concordance between TreeSHAP feature attributions and authoritative WHO/CDC diagnostic criteria). Finally, an interactive Clinical Decision Support System (CDSS) research prototype is implemented and validated, demonstrating calibrated differential candidate ranking, directional symptom evidence visualization, and robust safety guardrails.
```

---

### Step 7: Update and Re-package the Overleaf LaTeX Archive

* **Action**: Package all updated files into a clean Overleaf-ready ZIP archive:
  - Source directory: `Book_Writing/`
  - Output archive: `Book_Writing_Overleaf.zip`
  - Verification: Ensure all 7 chapter files, `rac.sty`, `main.tex`, `references.bib`, `FrontMatter/`, `Abstract/`, and all embedded figures in `figures/` are included with clean relative paths.

---

### Step 8: Update Repository Overview Documentation

* **Files**:
  - `Overview/research_consistency_overview.md`
  - `Overview/consistency_audit_report.md`
  - `Overview/pipeline_consistency_audit_report.md`
* **Action**: Synchronize all matrices to mark Phases 0 through 9 as **✅ 100% Completed, Verified, and Synchronized**.

---

## 4. Verification Checklist & Success Criteria

| Checkpoint | Target State | Verification Method |
| :--- | :--- | :--- |
| **Calibration Metrics in Ch 7** | RF ECE: $0.1114 \to 0.0575$, MLP: $0.2680 \to 0.0517$ | Match `research_work/reports/phase7_calibration_report.md` |
| **Calibration Figures in Ch 7** | `reliability_diagrams.png`, `calibration_metric_comparison.png` | Verified in `Book_Writing/figures/` |
| **TreeSHAP Metrics in Ch 7** | 72.7% WHO/CDC concordance, RF Jaccard: $0.5265$ | Match `research_work/reports/phase8_shap_report.md` |
| **TreeSHAP Figures in Ch 7** | `shap_clinical_concordance.png`, `shap_stability_comparison.png` | Verified in `Book_Writing/figures/` |
| **CDSS Prototype in Ch 7** | Documented FastAPI + UI architecture, 22 test cases | Match `prototype/` implementation |
| **Abstract Harmonization** | Noise, Calibration, SHAP, and Prototype claimed as completed | Textual inspection of `Book_Writing/Abstract/Abstract.tex` |
| **Overleaf Compilation** | 0 missing files, clean LaTeX compilation | Validated zip structure with relative paths |
| **Git Safety** | Zero remote git commands executed | Strictly local filesystem modifications |

---

## 5. Execution Command

When you are ready, say **"execute the remediation plan"** and I will implement this entire sequence end-to-end.
