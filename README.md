# Disease Prediction System Using Machine Learning
### A Robust, Explainable, and Uncertainty-Aware Framework Under Real-World Symptom Sparsity and Noise

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20XGBoost-orange.svg)](https://scikit-learn.org/)
[![Thesis](https://img.shields.io/badge/Degree-Master%20in%20Advanced%20Studies%20(MAS)-purple.svg)](#authorship--academic-affiliation)
[![LaTeX](https://img.shields.io/badge/Manuscript-LaTeX%20%7C%20Overleaf%20Ready-brightgreen.svg)](Book_Writing/)

---

## 📌 Executive Summary

Automated symptom-based disease prediction holds enormous potential for modern **Clinical Decision Support Systems (CDSS)**. However, contemporary literature evaluating public benchmarks (such as the Kaggle 41/42-class disease prediction dataset) routinely reports near-perfect or 100% classification accuracy without auditing underlying data integrity.

This repository presents the master codebase, empirical research pipeline, and 7-chapter Master's thesis for an independent forensic investigation and robust machine learning redesign:

1. **Forensic Audit of Benchmark Data**:
   - Disproved published Kaggle dataset claims: The dataset strictly contains **41 balanced disease classes** (120 samples each), not 42.
   - Uncovered **93.8% synthetic duplication** in `Training.csv` (only 304 unique symptom vectors across 4,920 records).
   - Uncovered **97.6% verbatim train-test leakage** (41 out of 42 testing vectors in `Testing.csv` are exact duplicate copies of training records), proving conventional 100% benchmark claims to be artifacts of data leakage and memorization.
2. **Three-Tier Evaluation Framework**:
   - **Protocol A (Standard Kaggle Split)**: Baseline replication under published split.
   - **Protocol B (Stratified 10-Fold CV)**: Evaluates random splitting across all samples.
   - **Protocol C (Duplicate-Aware Group 10-Fold CV)**: Groups identical symptom vectors strictly within isolated folds to achieve true zero-leakage out-of-sample evaluation.
3. **Model Generalization Breakdown**:
   - Under Protocol C, tree-based models suffer severe generalization collapses: **Decision Tree drops by $-17.44\%$** ($100\% \to 82.56\%$) and **XGBoost drops by $-12.80\%$** ($100\% \to 87.20\%$).
   - Additive and continuous models (**Logistic Regression, MLP, SVM**) retain 100% accuracy due to smooth evidence summation over all presenting symptoms.
4. **Leakage-Aware Feature Selection**:
   - Identifies a minimal consensus subset of **$k = 50$ symptoms** that preserves $\ge 95\%$ operational Macro-F1 retention relative to the full 132-symptom baseline.
5. **Clinical Symptom Sparsity Safeguard**:
   - Simulates real-world cold-start patient presentations ($s \in [1, 10]$ active symptoms vs. cohort natural mean of $7.45 \pm 3.59$).
   - Demonstrates that differential diagnostic candidate ranking (**Top-3 and Top-5**) provides a critical safeguard when single-label Top-1 accuracy degrades under sparse input vectors.

---

## 🔬 Benchmark Comparison Matrix

Evaluation across 9 algorithmic architectures across the three validation protocols:

| Model Architecture | Protocol A (Kaggle Split) | Protocol B (Stratified CV) | Protocol C (Duplicate-Aware Group CV) | Robustness Delta ($\Delta_{\text{Prot C - A}}$) |
| :--- | :---: | :---: | :---: | :---: |
| **Dummy (Uniform)** | 2.38% | 2.44% | 2.44% | $\pm 0.00\%$ |
| **Multinomial Naive Bayes** | 100.0% | 100.0% | 100.0% | $0.00\%$ |
| **Logistic Regression** | 100.0% | 100.0% | 100.0% | $0.00\%$ |
| **Decision Tree** | 97.62% | 100.0% | **82.56%** | **$-15.06\%$** |
| **Random Forest** | 97.62% | 100.0% | 100.0% | $+2.38\%$ |
| **Extra Trees** | 97.62% | 100.0% | 100.0% | $+2.38\%$ |
| **Support Vector Machine (RBF)** | 100.0% | 100.0% | 100.0% | $0.00\%$ |
| **Multi-Layer Perceptron (MLP)** | 100.0% | 100.0% | 100.0% | $0.00\%$ |
| **XGBoost** | 97.62% | 100.0% | **87.20%** | **$-10.42\%$** |

---

## 📂 Repository Architecture

```bash
Disease_Prediciton_System_ML/
├── Book_Writing/               # Complete 7-Chapter Master's Thesis (LaTeX source)
│   ├── main.tex                # Master document controller
│   ├── rac.sty                 # Thesis style package (customized for MAS NU format)
│   ├── references.bib          # Comprehensive BibTeX bibliography
│   ├── abbr.tex                # List of Abbreviations
│   ├── FrontMatter/            # Prelim pages (Cover, Title, Declaration, Acknowledgements)
│   ├── Abstract/               # Structured research abstract
│   ├── Chap1/ - Chap7/         # Individual thesis chapters (I through VII)
│   └── figures/                # Publication-quality figures & institutional crest
│
├── research_work/              # Empirical Python experiment suite
│   ├── src/                    # Core source code
│   │   ├── data_audit.py       # Forensic audit and anomaly detection
│   │   ├── data_loader.py      # Protocols A, B, and C splitters
│   │   ├── models.py           # Model definitions and hyperparameter configs
│   │   ├── run_benchmarks.py   # Multi-protocol benchmark execution engine
│   │   ├── feature_selection.py# Leakage-aware 5-algorithm feature selection
│   │   ├── sparsity_experiment.py # Clinical symptom sparsity simulator
│   │   └── utilities.py        # Logging, reporting, and serialization helpers
│   ├── reports/                # Numerical metrics, CSVs, and markdown audit reports
│   └── notebooks/              # Jupyter exploratory notebooks
│
├── data/                       # Datasets
│   ├── Training.csv            # 4,920 records x 133 columns (132 symptoms + prognosis)
│   └── Testing.csv             # 42 records x 133 columns
│
├── plan/                       # Pre-experiment scientific blueprint
│   └── research_plan.md        # Hypotheses (H1-H5), research questions, and methodology
│
├── Overview/                   # Consistency & traceability documentation
│   ├── README.md               # Folder manifest
│   └── research_consistency_overview.md # End-to-end plan-to-thesis mapping audit
│
├── requirements.txt            # Python dependencies
├── .gitignore                  # GitHub ignore configuration
└── README.md                   # Repository landing page
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Prerequisites & Environment Setup
Clone this repository and create an isolated Python virtual environment:

```bash
# Clone repository
git clone https://github.com/<your-username>/Disease_Prediciton_System_ML.git
cd Disease_Prediciton_System_ML

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Forensic Data Audit
Execute the forensic audit to inspect raw dimensions, verify duplicate rates, and confirm train-test leakage:

```bash
python research_work/src/data_audit.py
```
*Outputs: Generates tabular summaries and forensic reports in `research_work/reports/`.*

### 3. Run Multi-Protocol Benchmarks
Evaluate the 9 classifiers under Protocol A, Protocol B, and Protocol C:

```bash
python research_work/src/run_benchmarks.py
```
*Outputs: Evaluates all models, exports per-protocol confusion matrices, and writes `protocol_comparison.csv`.*

### 4. Run Leakage-Aware Feature Selection
Execute the 5-method feature selection audit (Chi-Square, Mutual Information, Random Forest Importance, L1-Logistic, and RFE) across symptom budgets $k \in \{132, 100, 70, 50, 30, 20, 10\}$:

```bash
python research_work/src/feature_selection.py
```

### 5. Run Clinical Symptom Sparsity Simulations
Simulate sparse patient presentation vectors ($s \in [1, 10]$ symptoms) and evaluate Top-1, Top-3, and Top-5 retrieval safeguards:

```bash
python research_work/src/sparsity_experiment.py
```

---

## 📖 Compiling the Thesis Manuscript

The complete manuscript is located in [`Book_Writing/`](Book_Writing/) and is prepared for compilation using **Overleaf** or any standard TeX distribution (TeX Live / MacTeX):

```bash
cd Book_Writing
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Thesis Structure
* **Chapter I**: Introduction, Research Motivation & Objectives
* **Chapter II**: Literature Review, Clinical Decision Support & Benchmark Vulnerabilities
* **Chapter III**: Forensic Audit of the Kaggle Disease Benchmark
* **Chapter IV**: Leakage-Aware Methodology & Experimental Framework
* **Chapter V**: Algorithmic Benchmark Results & Feature Selection Consensus
* **Chapter VI**: Clinical Symptom Sparsity & Robustness Analysis
* **Chapter VII**: Discussion, CDSS Translation Challenges & Future Roadmap

---

## 🎓 Authorship & Academic Affiliation

* **Author**: **Nasir uddin**
  * **Registration No**: `23880300014`
  * **Session**: `2023-2024`
  * **Program**: On-Campus Master in Advanced Studies (MAS) in Computer Science
  * **Institution**: Department of Computer Science, National University, Gazipur, Bangladesh
* **Supervisor**: **Husne Ara Rubaiyeat**
  * Assistant Professor, Department of Computer Science, National University, Bangladesh

---

## 📝 Citation

If you use this codebase, empirical findings, or the thesis manuscript in your academic research, please cite:

```bibtex
@mastersthesis{nasiruddin2025disease,
  author       = {Nasir uddin},
  title        = {Disease Prediction System Using Machine Learning: A Robust, Explainable, and Uncertainty-Aware Framework Under Real-World Symptom Sparsity and Noise},
  school       = {Department of Computer Science, National University},
  address      = {Gazipur, Bangladesh},
  year         = {2025},
  month        = {September},
  note         = {Thesis for the Degree of On-Campus Master in Advanced Studies (MAS) in Computer Science}
}
```

---

## 📄 License

This research codebase and accompanying documentation are distributed under the **MIT License**. See [LICENSE](LICENSE) for details. Academic usage, extension, and replication are warmly encouraged with appropriate attribution.
