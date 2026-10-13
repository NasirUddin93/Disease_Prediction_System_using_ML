# Research Pipeline Consistency Audit: Plan $\to$ Research Work $\to$ Book Writing

**Project Title**: *Disease Prediction System Using Machine Learning: A Robust, Explainable, and Uncertainty-Aware Framework Under Real-World Symptom Sparsity and Noise*  
**Degree**: Master in Advanced Studies (MAS) in Computer Science  
**Department**: Department of Computer Science, National University, Gazipur, Bangladesh  
**Candidate**: Nasir uddin (Registration No: `23880300014`, Session: `2023-2024`)  
**Supervisor**: Husne Ara Rubaiyeat, Assistant Professor  
**Audit Date**: October 2026 (Thesis Submission Target: September 2025 / Session 2023–2024)  

---

## 1. Executive Summary

This document establishes the formal end-to-end consistency, numerical verification, and traceability audit across the three core layers of the Master's research suite:

1. **`plan/` (Scientific Blueprint & Execution Plans)**:
   * [`research_plan.md`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/plan/research_plan.md): Foundational blueprint, Research Questions (RQ1–RQ5), and Pre-Experiment Hypotheses (H1–H5).
   * [`phase_05_completion_plan.md`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/plan/phase_05_completion_plan.md): Specification for controlled clinical symptom sparsity stress testing.
   * [`phase_06_execution_plan.md`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/plan/phase_06_execution_plan.md): Specification for bidirectional stochastic symptom reporting noise and error asymmetry.
   * [`phase_07_execution_plan.md`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/plan/phase_07_execution_plan.md): Methodological proposal for post-hoc probability calibration and uncertainty quantification.

2. **`research_work/` (Empirical Code, Datasets, and Artifacts)**:
   * [`src/`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/research_work/src): Python execution suite (data audit, cross-validation splitters, baseline benchmarking, feature selection, sparsity masking, noise perturbation, calibration).
   * [`reports/`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/research_work/reports): Verifiable CSV result files, JSON summaries, Markdown verification reports, and high-DPI figure plots.

3. **`Book_Writing/` (The Complete 7-Chapter LaTeX Thesis)**:
   * Formatted using the institutional Rackham / National University MAS format (`rac.sty`, `FrontMatter/`, `Abstract/`, `references.bib`, and Chapters I through VII).

```
┌─────────────────────────────────┐      ┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│              plan/              │ ───► │         research_work/          │ ───► │          Book_Writing/          │
│   Hypotheses, RQs, Blueprints   │      │   Python Code, CSVs, Figures    │      │   7-Chapter LaTeX Thesis Book   │
└─────────────────────────────────┘      └─────────────────────────────────┘      └─────────────────────────────────┘
```

---

## 2. Global Pipeline Alignment & Traceability Matrix

| Phase & Focus | `plan/` Blueprint | `research_work/` Status | `Book_Writing/` Manifest | Alignment Verdict |
| :--- | :--- | :--- | :--- | :---: |
| **Phase 0: Dataset Forensics (`E1`)** | Section 5 of [`research_plan.md`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/plan/research_plan.md) | [`data_audit.py`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/research_work/src/data_audit.py)<br>[`dataset_forensic_report.md`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/research_work/reports/dataset_forensic_report.md) | Chapter III (Table 3.1, Figs 3.1–3.4) | ✅ **100% Verified** |
| **Phase 1: Literature Review** | Section 2 & 13 | Cross-referenced citations | Chapter II (Comprehensive Related Work) | ⚠️ **Minor Gap** *(Reporting noise omitted in Section 2.5)* |
| **Phase 2 & 3: Multi-Protocol Hierarchy (`E2`, `E3`)** | Sections 6 & 7 (H1, RQ1) | [`run_benchmarks.py`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/research_work/src/run_benchmarks.py)<br>[`protocol_comparison.csv`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/research_work/reports/protocol_comparison.csv) | Chapter IV (Sec 4.1) & Chapter V (Table 5.1, Figs 5.1–5.2) | ✅ **100% Verified** |
| **Phase 4: Feature Selection (`E4`)** | Section 8 | [`feature_selection.py`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/research_work/src/feature_selection.py)<br>[`phase4_final_summary.csv`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/research_work/reports/phase4_final_summary.csv) | Chapter V (Sec 5.2, Table 5.2, Figs 5.3–5.5) | ✅ **100% Verified** |
| **Phase 5: Symptom Sparsity (`E5`)** | Section 9.1 & [`phase_05_completion_plan.md`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/plan/phase_05_completion_plan.md) (H2, RQ2) | [`sparsity_experiment.py`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/research_work/src/sparsity_experiment.py)<br>[`sparsity_summary.csv`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/research_work/reports/sparsity_summary.csv) | Chapter VI (Table 6.1, Fig 6.1) | ✅ **100% Verified** |
| **Phase 6: Reporting Noise (`E6`)** | Section 9.2 & [`phase_06_execution_plan.md`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/plan/phase_06_execution_plan.md) (H3, RQ3) | [`noise_experiment.py`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/research_work/src/noise_experiment.py)<br>[`noise_summary.csv`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/research_work/reports/noise_summary.csv) | Chapter VII (Sec 7.2, Table 7.1, Figs 7.1–7.2) | ⚠️ **Completed in Code & Thesis** *(Contradicted in Abstract)* |
| **Phase 7: Probability Calibration (`E7`)** | Section 10 & [`phase_07_execution_plan.md`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/plan/phase_07_execution_plan.md) (H4, RQ4) | [`calibration_experiment.py`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/research_work/src/calibration_experiment.py)<br>Pilot 1-fold run in [`calibration_summary.csv`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/research_work/reports/calibration_summary.csv) | Chapter VII (Sec 7.5.1 Future Work) | ⚠️ **Contradiction with Chapter I** *(Claimed as contribution in Ch 1)* |
| **Phase 8: Explainable AI / SHAP (`E8`)** | Section 11 (H5, RQ5) | Not yet executed in `research_work/` | Chapter VII (Sec 7.5.2 Future Work) | ⚠️ **Contradiction with Chapter I** *(Claimed as contribution in Ch 1)* |
| **Phase 9: CDSS Prototype (`E9`)** | Section 12 | Not yet executed | Chapter VII (Sec 7.5.3 Future Work) | ✅ **Consistent Roadmap** |

---

## 3. Empirical Quantitative Verification

Every single metric reported across the 7 chapters of the thesis was cross-checked against raw execution files:

### 3.1 Phase 0: Forensic Audit of the Benchmark Dataset
* **Published vs. Empirical Dimensions**:
  * Kaggle documentation claimed 42 diseases; the dataset strictly contains **41 balanced disease classes** (120 records per class in `Training.csv`).
  * In `Testing.csv`, 40 diseases have 1 sample, and 1 disease (*Fungal infection*) has 2 samples ($40 + 2 = 42$ records).
  * 1 empty artifact column (`Unnamed: 133`) successfully pruned.
* **Redundancy & Duplication**:
  * Out of 4,920 records in `Training.csv`, exactly **304 unique symptom vectors** exist ($\mathbf{93.82\%}$ duplicate rows).
  * Each disease consists of only 5 to 10 unique symptom patterns replicated 12 to 24 times.
* **Train-Test Leakage**:
  * **41 out of 42 testing records (97.62%)** are verbatim duplicate copies of training records.
  * *Thesis Match*: Perfectly matches [Chapter III Table 3.1](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/Book_Writing/Chap3/Chapter3.tex#L6-L27).

### 3.2 Phase 2 & 3: Multi-Protocol Benchmark & Leakage Isolation
* **Protocol Comparison (Table 5.1)**:
  * **Protocol A (Kaggle Split)**: Models achieve near-perfect memorization (97.6%–100.0%).
  * **Protocol B (Stratified 10-Fold CV)**: Tree-based models achieve artificial 100.0% accuracy due to duplicate vectors straddling folds.
  * **Protocol C (Duplicate-Aware Group 10-Fold CV)**:
    * Single Decision Tree collapses from **100.0% to 82.56%** (Macro-F1: 0.8300).
    * XGBoost collapses from **100.0% to 87.20%** (Macro-F1: 0.8748).
    * Extra Trees drops to **97.20%** (Macro-F1: 0.9721).
    * Random Forest retains **99.39%** accuracy.
    * Additive models (Logistic Regression, MLP, SVM) retain **100.0%**.
  * *Thesis Match*: Perfectly matches [Chapter V Table 5.1](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/Book_Writing/Chap5/Chapter5.tex#L9-L31) and [`protocol_comparison.csv`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/research_work/reports/protocol_comparison.csv).

### 3.3 Phase 4: Leakage-Aware Feature Selection ($k \in \{132 \to 10\}$)
* **Operational Criterion**: Smallest subset retaining $\ge 95.0\%$ Macro-F1 relative to baseline under Protocol C.
* **Empirical Subset ($k=50$)**:
  * Mutual Information: Logistic Regression (96.9% retained), MLP (97.7% retained), Random Forest (95.3% retained).
  * $L_1$ Logistic: Logistic Regression (96.7% retained), MLP (97.0% retained), Random Forest (93.2% retained).
* **Chi-Square Anomaly**: Chi-Square ranking fails catastrophically ($35.4\%$ F1 at $k=50$) because it penalizes shared combinatorial symptoms (e.g., *vomiting*, *fatigue* occurring in 17 diseases; correlation $r = -0.9069$ with prevalence).
* *Thesis Match*: Perfectly matches [Chapter V Table 5.2](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/Book_Writing/Chap5/Chapter5.tex#L78-L110) and [`phase4_final_summary.csv`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/research_work/reports/phase4_final_summary.csv).

### 3.4 Phase 5: Symptom Sparsity Stress Testing ($s \in [1, 10]$)
* **Cohort Natural Baseline**: $7.45 \pm 3.59$ active symptoms (range: 3–17).
* **Single-Label Collapse vs. Multi-Candidate Differential Buffer**:
  * At $s = 3$ symptoms (Full 132 space):
    * Logistic Regression: Top-1 Acc = $0.758 \pm 0.079$, Macro-F1 = $0.551 \pm 0.045$, **Top-3 Acc = 0.885**, **Top-5 Acc = 0.934**.
    * Random Forest: Top-1 Acc = $0.767 \pm 0.100$, Macro-F1 = $0.579 \pm 0.066$, **Top-3 Acc = 0.895**, **Top-5 Acc = 0.940**.
    * MLP: Top-1 Acc = $0.735 \pm 0.114$, Macro-F1 = $0.583 \pm 0.069$, **Top-3 Acc = 0.883**, **Top-5 Acc = 0.927**.
  * *Thesis Match*: Perfectly matches [Chapter VI Table 6.1](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/Book_Writing/Chap6/Chapter6.tex#L25-L84) and [`sparsity_summary.csv`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/research_work/reports/sparsity_summary.csv).

### 3.5 Phase 6: Stochastic Reporting Noise & Asymmetry ($N = 20,520$ Runs)
* **Testing Hypothesis H3 Under Matched-Error-Burdens ($\sim 1.49$ Altered Features)**:
  * 20% Dropout (1.49 omitted features): Logistic Regression F1 = $0.9420 \pm 0.0145$, Random Forest F1 = $0.9113 \pm 0.0296$.
  * Matched 1.20% False Positive (1.49 spurious features): Logistic Regression F1 = $0.9988 \pm 0.0025$, Random Forest F1 collapses to $0.7743 \pm 0.0352$.
  * *Verdict*: H3 is **partially supported** and architecture-dependent. Spurious activations severely degrade tree-partitioning models (Random Forest, XGBoost), whereas linear and neural models degrade more under omission.
* *Thesis Match*: Perfectly matches [Chapter VII Table 7.1](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/Book_Writing/Chap7/Chapter7.tex#L60-L95) and [`noise_summary.csv`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/research_work/reports/noise_summary.csv).

---

## 4. Identified Discrepancies & Required Action Items

| ID | Issue Description | Location in Code / Manuscript | Recommended Action |
| :---: | :--- | :--- | :--- |
| **D1** | **Abstract relegates Phase 6 (Noise) to future roadmap** | [`Abstract/Abstract.tex`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/Book_Writing/Abstract/Abstract.tex#L1) | **Update Abstract**: Formally include completed Phase 6 noise results (20,520 runs, error asymmetry), restricting the future roadmap strictly to Calibration and TreeSHAP. |
| **D2** | **Chapter I claims Calibration & SHAP as completed contributions** | [`Chap1/Chapter1.tex`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/Book_Writing/Chap1/Chapter1.tex#L31-L32) | **Align Contributions 4 & 5**: Reframe them around multi-candidate differential ranking buffers under sparsity/noise and dataset-level algorithmic feature stability, rather than claiming completed SHAP or full empirical ECE. |
| **D3** | **Chapter II Research Gaps omit reporting noise** | [`Chap2/Chapter2.tex`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/Book_Writing/Chap2/Chapter2.tex#L25-L29) | **Add Noise Gap**: Add a fourth gap in Section 2.5 addressing bidirectional symptom misreporting and matched-error-burden sensitivity. |
| **D4** | **Overview metadata previously referenced wrong institution** | [`Overview/research_consistency_overview.md`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/Overview/research_consistency_overview.md#L4-L7) | **Harmonize Metadata**: Update overview documentation with official National University MAS thesis metadata (National University, Gazipur; Supervisor Husne Ara Rubaiyeat). |
| **D5** | **Table 7.1 caption lacks explicit representation label** | [`Chap7/Chapter7.tex`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/Book_Writing/Chap7/Chapter7.tex#L61) | **Clarify Caption**: Explicitly mention in Table 7.1 caption that metrics correspond to the reference `Full 132-Symptom Space`. |

---

## 5. Structural Map: Plan vs. Thesis Chapters

```
Proposed Blueprint (plan/research_plan.md)         Final Thesis (Book_Writing/)
─────────────────────────────────────────────     ─────────────────────────────────────────────────────────────
Chapter 1: Introduction                       ──► Chapter I:   Introduction
Chapter 2: Literature Review                  ──► Chapter II:  Literature Review
Chapter 3: Forensics & Methodology            ──► Chapter III: Dataset Forensic Analysis
Chapter 4: Algorithmic Benchmarking           ──► Chapter IV:  Methodology, Experimental Design, Hypotheses
Chapter 5: Robustness (Sparsity + Noise)      ──► Chapter V:   Benchmark Results & Feature Selection (Phases 2, 3, 4)
Chapter 6: Calibration & Explainability (XAI) ──► Chapter VI:  Clinical Symptom Sparsity Analysis (Phase 5)
Chapter 7: CDSS Prototype                     ──► Chapter VII: Discussion, Conclusion, & Future Work
Chapter 8: Discussion & Conclusion                 - Section 7.2: Completed Phase 6 Noise Benchmark
                                                   - Section 7.5: Future Work (Calibration, SHAP, Prototype)
```

---

## 6. Conclusion

The pipeline from **Plan $\to$ Empirical Research $\to$ Book Writing** demonstrates high scientific integrity, zero numerical contradictions, and rigorous cross-validation isolation. Addressing the editorial refinements noted in Section 4 will ensure 100% internal narrative consistency across the thesis manuscript.
