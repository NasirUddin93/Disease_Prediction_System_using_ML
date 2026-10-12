# Disease Prediction System Using Machine Learning: Research Suite

**Subtitle**: *A Robust, Explainable and Uncertainty-Aware Machine Learning Framework for Symptom-Based Multi-Disease Prediction*  
**Master's Thesis Research Project**

---

## Directory Architecture

```
Disease_Prediciton_System_ML/
├── plan/
│   └── research_plan.md          # Comprehensive Master's research blueprint
│
├── research/
│   ├── data/
│   │   ├── raw/                  # Downloaded CSV files (Training.csv, Testing.csv)
│   │   └── processed/            # Sanitized, deduplicated, and partitioned splits
│   │
│   ├── notebooks/
│   │   └── 01_dataset_forensics.ipynb  # Interactive forensic exploratory notebook
│   │
│   ├── src/
│   │   ├── data_audit.py         # Complete forensic analysis & metrics extraction
│   │   ├── preprocessing.py      # Sanitization, splitting (Protocols A, B, C)
│   │   └── utilities.py          # Logging, export, and I/O helpers
│   │
│   ├── reports/
│   │   ├── dataset_forensic_report.md  # Detailed forensic documentation
│   │   ├── dataset_summary.csv         # Machine-readable tabular summary
│   │   ├── class_distribution.csv      # Per-class sample counts and pattern metrics
│   │   └── figures/                    # Publication-grade figures (300 DPI)
│   │
│   └── README.md
│
└── Paper_Writing/                # LaTeX research manuscript ready for Overleaf
    ├── main.tex                  # Primary LaTeX manuscript
    ├── references.bib            # BibTeX literature citations
    └── figures/                  # Vector / High-DPI figures embedded in paper
```

---

## Execution Order
1. **Phase 0: Dataset Forensics (`E1`)**: Rigorous audit of raw data dimensions, class balance, duplicates, and train-test leakage.
2. **Phase 1: Literature Review & Gap Analysis**: Formalizing hypotheses against existing studies.
3. **Phase 2: Leakage-Aware Data Pipeline**: Protocol A (Kaggle), Protocol B (Stratified CV), Protocol C (Duplicate-Aware CV).
4. **Phase 3: Baseline ML Benchmark**: Evaluating model hierarchy (Baselines $\to$ Ensembles $\to$ Kernel $\to$ Neural).
5. **Phase 4: Feature Selection**: Measuring performance across $\{132 \to 10\}$ symptoms.
6. **Phase 5: Sparsity & Noise Robustness**: Cold-start presentations and reporting perturbations.
7. **Phase 6: Uncertainty & Probability Calibration**: ECE, Brier score, Reliability curves.
8. **Phase 7: Explainable AI (SHAP)**: Global/local attributions & medical guideline alignment.
9. **Phase 8: Decision Support Prototype**: Interactive clinical research interface.
10. **Thesis & Paper Writing**: Overleaf LaTeX compilation.
