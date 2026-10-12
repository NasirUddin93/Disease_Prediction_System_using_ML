# Research Pipeline Consistency & Project Overview

**Project Title**: *Disease Prediction System Using Machine Learning: A Robust, Explainable, and Uncertainty-Aware Framework Under Real-World Symptom Sparsity and Noise*  
**Author**: Nasir Uddin (ID: 243043)  
**Supervisor**: Dr. Rashed Mazumder, Associate Professor  
**Program**: Professional Masters in Information Technology (PMIT), Institute of Information Technology (IIT), Jahangirnagar University  
**Date**: October 2026  

---

## 1. Executive Summary

This document establishes the end-to-end traceability, verification, and consistency audit across the three core layers of the Master's research suite:
1. **`plan/research_plan.md`**: The foundational scientific blueprint, research questions (RQ1–RQ5), and pre-experiment hypotheses (H1–H5).
2. **`research_work/`**: The empirical Python execution suite, containing data audit routines, cross-validation splitters, benchmark trainers, and CSV/figure artifacts.
3. **`Book_Writing/`**: The complete Master's project report / book written in LaTeX (using the institutional `rac.sty` Rackham thesis format).

```
┌───────────────────────────┐      ┌───────────────────────────┐      ┌───────────────────────────┐
│   plan/research_plan.md   │ ───► │      research_work/       │ ───► │       Book_Writing/       │
│  (Hypotheses & Blueprint) │      │  (Code, Data, & Reports)  │      │  (7-Chapter LaTeX Thesis) │
└───────────────────────────┘      └───────────────────────────┘      └───────────────────────────┘
```

---

## 2. Global Pipeline Alignment Matrix

| Phase & Experiment ID | Focus & Scientific Scope | `plan/research_plan.md` | `research_work/` Status | `Book_Writing/` Manifest | Alignment Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Phase 0: Dataset Forensics (`E1`)** | Data dimensions, 41 vs. 42 classes, duplicate rows, train-test leakage | Planned in Section 5 | `data_audit.py`, `dataset_forensic_report.md` | Chapter III (Forensic Analysis, Table 3.1, Figs 3.1–3.4) | ✅ **Fully Aligned** |
| **Phase 1: Literature Review** | Medical CDSS, ML baselines, XAI, and benchmark leakage literature | Planned in Section 2 & 13 | Cross-referenced citations | Chapter II (Comprehensive Related Work) | ✅ **Fully Aligned** |
| **Phase 2: Validation Protocols (`E3`)** | Protocol A (Kaggle), Protocol B (CV), Protocol C (Duplicate-Aware Group CV) | Formulated in Section 6 | `data_loader.py` (splits data into zero-leakage groups) | Chapter IV (Section 4.1) & Chapter V (Results) | ✅ **Fully Aligned** |
| **Phase 3: Model Hierarchy (`E2`)** | 9-model evaluation (Baselines, Ensembles, SVM, MLP) | Formulated in Section 7 | `run_benchmarks.py`, `protocol_comparison.csv` | Chapter V (Table 5.1, Figs 5.1–5.2) | ✅ **Fully Aligned** |
| **Phase 4: Feature Selection (`E4`)** | Dimensionality reduction across $\{132 \to 10\}$ symptoms; $\ge 95\%$ F1 retention | Formulated in Section 8 | `feature_selection.py`, `phase4_final_summary.csv` | Chapter V (Table 5.2 with LogReg, MLP, RF; Figs 5.3–5.5) | ✅ **Fully Aligned** |
| **Phase 5: Symptom Sparsity (`E5`)** | Cold-start chief complaint restriction ($s \in [1, 10]$ active symptoms) | Formulated in Section 9.1 | `sparsity_experiment.py`, `sparsity_summary.csv` | Chapter VI (Table 6.1, Fig 6.1, differential buffer) | ✅ **Reconciled (Provisional)** |
| **Phase 6: Reporting Noise (`E6`)** | False-negative dropout & false-positive misreporting ($0\% \to 30\%$) | Formulated in Section 9.2 (H3) | Not yet executed | Chapter IV (Formulation) & Chapter VII (Future Work) | ✅ **Consistent Roadmap** |
| **Phase 7: Calibration (`E7`)** | ECE, Brier score, Platt scaling, Isotonic reliability diagrams | Formulated in Section 10 (H4) | Not yet executed | Chapter IV (Formulation) & Chapter VII (Future Work) | ✅ **Consistent Roadmap** |
| **Phase 8: Explainable AI (`E8`)** | TreeSHAP attributions & WHO/CDC diagnostic guideline audit | Formulated in Section 11 (H5) | Not yet executed | Chapter IV (Formulation) & Chapter VII (Future Work) | ✅ **Consistent Roadmap** |
| **Phase 9: CDSS Prototype (`E9`)** | Interactive FastAPI/web interface with differential diagnostic ranking | Formulated in Section 12 | Not yet implemented | Chapter VII (Section 7.4 Deployment Roadmap) | ✅ **Consistent Roadmap** |

---

## 3. Structural Mapping: Plan vs. Thesis Chapters

The initial blueprint (`plan/research_plan.md`) outlined an 8-chapter structure under the premise that all 9 experimental phases would be fully executed before writing. Because Phases 6–9 are currently unexecuted in `research_work/`, the manuscript in `Book_Writing/` appropriately consolidates the work into **7 Chapters**, routing unexecuted experiments to Future Work to preserve scientific integrity:

```
Proposed Plan (8 Chapters)                      Actual Manuscript (7 Chapters)
─────────────────────────────────             ─────────────────────────────────────────
Chapter 1: Introduction                       ──► Chapter I:   Introduction
Chapter 2: Literature Review                  ──► Chapter II:  Literature Review
Chapter 3: Forensics & Methodology            ──► Chapter III: Dataset Forensic Analysis
Chapter 4: Algorithmic Benchmarking           ──► Chapter IV:  Methodology & Experimental Design
Chapter 5: Robustness (Sparsity + Noise)      ──► Chapter V:   Benchmark Results & Feature Selection
Chapter 6: Calibration & Explainability (XAI) ──► Chapter VI:  Clinical Symptom Sparsity Analysis
Chapter 7: CDSS Prototype                     ──► Chapter VII: Discussion, Conclusion, & Future Work
Chapter 8: Discussion & Conclusion                 (Integrates Noise, Calibration, SHAP, and Prototype)
```

---

## 4. Phase-by-Phase Verification & Empirical Findings

### Phase 0: Forensic Audit of the Benchmark
* **Kaggle Published Claims Disproven**: While online Kaggle documentation lists 42 disease classes, the dataset strictly contains **41 balanced classes** (120 training samples each).
* **Extreme Synthetic Duplication**: Across 4,920 records in `Training.csv`, there are only **304 unique symptom vectors** (an extreme duplication rate of **93.8%**).
* **Verbatim Test Set Leakage**: Out of 42 testing records in `Testing.csv`, **41 (97.6%)** are verbatim duplicate copies of training records, invalidating standard accuracy claims reported in prior literature.

### Phase 2 & 3: Leakage-Aware Multi-Model Benchmarking
* **Protocol A (Kaggle Official Split)**: All conventional models report near-perfect accuracy (97.6%–100.0%) due to memorization of duplicated test vectors.
* **Protocol B (Stratified 10-Fold CV)**: All tree ensembles achieve artificial 100.0% accuracy because duplicate symptom vectors straddle training and validation folds.
* **Protocol C (Duplicate-Aware Group 10-Fold CV)**: Partitioning identical symptom vectors into isolated folds exposes dramatic generalization drops in partitioned tree models:
  * **Decision Tree**: Accuracy collapses from **100.0% to 82.56%** ($-17.44\%$).
  * **XGBoost**: Accuracy collapses from **100.0% to 87.20%** ($-12.80\%$).
  * **Additive & Neural Models (Logistic Regression, MLP, SVM)**: Retain 100.0% accuracy due to smooth evidence summation over all reported features.

### Phase 4: Feature Selection & Dimensionality Reduction
* **Operational Criterion**: The minimal symptom subset retaining $\ge 95.0\%$ Macro-F1 relative to the full 132-feature baseline under Protocol C.
* **Minimal Consensus Subset ($k=50$)**:
  * **Mutual Information**: Logistic Regression (96.9% retained F1), MLP (97.7% retained F1), Random Forest (95.3% retained F1).
  * **$L_1$-Regularized Logistic**: Logistic Regression (96.7% retained F1), MLP (97.0% retained F1), Random Forest (93.2% retained F1).
* **Chi-Square Anomaly**: Chi-Square ranking severely degraded performance (35.4% F1 at $k=50$) because it over-ranked ultra-specific single-disease symptoms while discarding high-information shared symptoms (e.g., *vomiting*, *fatigue*) necessary to differentiate overlapping diseases.

### Phase 5: Symptom Sparsity Stress Test
* **Cohort Natural Distribution**: Patients in the dataset present $7.45 \pm 3.59$ active symptoms (range: 3–17).
* **Evaluated Budgets**: Controlled masking evaluated across $s \in \{\text{Full}, 10, 8, 6, 5, 4, 3, 2, 1\}$ presenting chief complaints.
* **Differential Diagnostic Buffer**: While single-label Top-1 accuracy collapses steeply below $s=5$ (dropping to 73.6%–76.5% at $s=3$ and 27.9%–50.4% at $s=1$), multi-candidate **Top-3 accuracy remains $\ge 88.6\%$** and **Top-5 accuracy remains $\ge 93.0\%$** at $s=3$.
* **Provisional Status**: Table 6.1 and Figure 6.1 are explicitly marked as provisional pending completion of the full model-by-budget-by-seed experimental matrix.

---

## 5. Critical Editorial & Methodological Reconciliations Applied

During recent peer-review audits, five critical inconsistencies were identified and formally resolved:

1. **Title Unification**:
   * *Issue*: `main.tex` had a short title, while `Declaration.tex` and `Certificate.tex` used the full thesis title.
   * *Resolution*: Unified the title across all three documents:  
     `"Disease Prediction System Using Machine Learning: A Robust, Explainable, and Uncertainty-Aware Framework Under Real-World Symptom Sparsity and Noise"`.

2. **Table 5.2 Completeness (MLP Data Restored)**:
   * *Issue*: Section 5.2.1 cited MLP achieving 97.7% retained F1 at $k=50$, but Table 5.2 only listed Logistic Regression and Random Forest.
   * *Resolution*: Expanded Table 5.2 to a 14-column layout presenting Logistic Regression, MLP Neural Net, and Random Forest across all subsets ($k \in \{132 \to 10\}$).

3. **Table 6.1 Column Swap Fixed**:
   * *Issue*: Top-1 Accuracy (`0.759`) and Macro-F1 (`0.553`) were swapped in earlier PDF renderings, displaying `0.553` under Top-1.
   * *Resolution*: Explicitly separated `\textbf{Top-1 Acc}` and `\textbf{Macro-F1}` in Table 6.1 and updated Section 6.2 prose to cite both metrics together.

4. **Scoping Abstract Claims vs. Future Work**:
   * *Issue*: The Abstract previously claimed noise modeling, calibration, and TreeSHAP as completed results, conflicting with Chapter VII.
   * *Resolution*: Updated the Abstract to strictly claim completed empirical phases (forensics, protocols, feature selection, sparsity), presenting noise, calibration, and SHAP as the planned research roadmap.

5. **Dataset-Level Evidence vs. Clinical Validation**:
   * *Issue*: Earlier drafts claimed feature subsets were "clinically validated".
   * *Resolution*: Replaced all such claims with *"dataset-level algorithmic stability and cross-method agreement on this benchmark"*, explicitly noting that external clinical validation requires prospective patient cohort studies.

---

## 6. Overleaf Artifact Archive

The entire LaTeX book project has been packaged and verified for direct Overleaf upload:
* **Archive Path**: `/home/nasir-uddin/Data/MAS/Disease_Prediciton_System_ML/Book_Writing_Overleaf.zip`
* **File Count**: 39 files (complete with `main.tex`, `rac.sty`, `references.bib`, `FrontMatter/`, `Abstract/`, all 7 chapter tex files, high-DPI figures, and university logos).
* **Compilation Status**: Zero missing files, clean pdflatex compilation, and free of blocking `latexmkrc` flags.
