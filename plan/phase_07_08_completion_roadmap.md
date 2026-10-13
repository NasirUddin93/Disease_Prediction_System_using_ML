# Phase 07 & 08 Completion and Validation Roadmap

**Document Status**: Active Implementation Roadmap  
**Project**: *Disease Prediction System Using Machine Learning: A Robust, Explainable, and Uncertainty-Aware Framework Under Real-World Symptom Sparsity and Noise*  
**Candidate**: Nasir uddin (Registration No: `23880300014`, Session: `2023-2024`)  
**Supervisor**: Husne Ara Rubaiyeat, Assistant Professor  

---

## 1. Context & Motivation

A recent audit of the public repository and thesis manuscript highlighted that:
1. **Phase 07 (Probability Calibration)** code was committed to GitHub, but the resulting CSV summaries reflected a preliminary single-fold run (`n_folds = 1`, `sd = 0.0`) instead of the planned 10-fold Protocol C cross-validation benchmark.
2. **Phase 08 (Explainable AI / SHAP)** was formulated in the research blueprint and cited in future work, but has not yet been executed in code.
3. The repository documentation (`Overview/research_consistency_overview.md`) still marks Phase 07 and 08 as uncompleted.

This roadmap outlines the systematic four-stage execution sequence to bring both phases to full completion, verification, and publication readiness.

---

## 2. Four-Stage Execution Roadmap

```
┌─────────────────────────────────┐
│ Stage 1: Phase 07 10-Fold Run   │ ──► Execute full 10-fold Protocol C calibration benchmark
└─────────────────────────────────┘     Verify ECE, Brier, and Log-loss across all folds
                 │
                 ▼
┌─────────────────────────────────┐
│ Stage 2: Phase 08 Execution     │ ──► Implement research_work/src/shap_experiment.py
└─────────────────────────────────┘     Compute TreeSHAP & LinearSHAP attributions
                 │                      Quantify explanation stability & WHO/CDC concordance
                 ▼
┌─────────────────────────────────┐
│ Stage 3: Documentation & Thesis │ ──► Update Overview/research_consistency_overview.md
└─────────────────────────────────┘     Update Overview/consistency_audit_report.md
                 │                      Reflect completed findings in Chapter VII
                 ▼
┌─────────────────────────────────┐
│ Stage 4: Git Synchronization    │ ──► Commit complete artifacts (CSVs, figures, reports)
└─────────────────────────────────┘     Push clean, verified state to GitHub master
```

---

## 3. Detailed Stage Breakdown

### Stage 1: Phase 07 Complete 10-Fold Benchmark
* **Command**:
  ```bash
  python3 research_work/src/calibration_experiment.py --max-folds 10
  ```
* **Expected Outputs**:
  * `research_work/reports/calibration_fold_results.csv`: Exactly 1,080 fold rows (10 folds $\times$ 2 representations $\times$ 9 models $\times$ 3 methods $\times$ 2 core baseline conditions + robustness sweeps).
  * `research_work/reports/calibration_summary.csv`: 108 summary rows with true `n_folds = 10`, genuine `_mean` and non-zero `_sd` for ECE, Brier score, Log loss, and Top-$k$ accuracies.
  * `research_work/reports/calibration_oof_probabilities.csv`: Complete out-of-fold probability vectors for all 4,920 records.
  * `research_work/reports/figures/reliability_diagrams.png`: 10-fold publication-grade reliability diagrams.
  * `research_work/reports/phase7_calibration_report.md`: Markdown report detailing empirical confirmation status of Hypothesis H4.

---

### Stage 2: Phase 08 Explainable AI Suite Implementation & Run
* **Execution Blueprint**: [`plan/phase_08_execution_plan.md`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/plan/phase_08_execution_plan.md)
* **Code Implementation**: `research_work/src/shap_experiment.py`
  - Protocol C 10-fold model fitting for Random Forest, Extra Trees, Decision Tree, Logistic Regression, and MLP.
  - TreeSHAP and LinearSHAP calculation.
  - Cross-fold explanation stability evaluation (Jaccard index and Spearman rank correlation of top features).
  - Automated concordance comparison against WHO / CDC diagnostic symptom definitions across 6 reference diseases (Dengue, Malaria, Typhoid, Asthma, Diabetes, GERD).
* **Outputs**:
  - `research_work/reports/shap_global_importance.csv`
  - `research_work/reports/shap_fold_stability.csv`
  - `research_work/reports/clinical_concordance_audit.csv`
  - `research_work/reports/figures/shap_summary_top_diseases.png`
  - `research_work/reports/phase8_shap_report.md`

---

### Stage 3: Documentation & Thesis Alignment
1. **Overview Documentation Update**:
   - Update [`Overview/research_consistency_overview.md`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/Overview/research_consistency_overview.md) alignment matrix: mark Phase 07 and Phase 08 as **✅ Completed & Verified**.
   - Harmonize candidate and supervisor metadata in `Overview/`.
2. **Thesis Manuscript Update**:
   - In [`Book_Writing/Chap7/Chapter7.tex`](file:///home/nasir-uddin/data/MAS/Disease_Prediction_System_using_ML/Book_Writing/Chap7/Chapter7.tex), move Calibration (H4) and SHAP (H5) from prospective Future Work into completed empirical sections or dedicated subsections.
   - Embed generated reliability diagrams and SHAP summary figures.

---

### Stage 4: Version Control Synchronization
* Stage all updated CSVs, high-resolution figures, Markdown reports, and thesis sources:
  ```bash
  git add research_work/reports/
  git add research_work/src/
  git add plan/
  git add Overview/
  git add Book_Writing/
  git commit -m "Complete Phase 07 10-fold calibration and Phase 08 SHAP explainability benchmarks"
  git push origin master
  ```
