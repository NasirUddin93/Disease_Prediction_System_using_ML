# Lightweight Overleaf Compilation Guide

This guide details optimizations to ensure the **Disease Prediction System** thesis compiles quickly and reliably on Overleaf's free tier without exceeding compile timeout limits.

---

## 1. Down-scale High-DPI Raster Images
High-resolution figures (e.g. 300+ DPI, >4000px dimensions) consume extensive memory and processing time during pdflatex rasterization.
- Figures in `figures/` should be resized to ~150 DPI target resolution:
  - `NU_logo.png`: Resized from 2000x2000 to 600x600 px (~130 KB).
  - `noise_degradation_curves.png`: Resized from 4500x1650 to 2250x825 px (~309 KB).
  - `noise_type_comparison.png`: Resized from 4500x1650 to 2250x825 px (~186 KB).
  - `sparsity_decay_curves.png`: Resized from 5400x1650 to 2700x825 px (~398 KB).
- Total `figures/` directory size drops from ~4.8 MB to ~3.7 MB, substantially cutting PDF compilation overhead.

---

## 2. Draft Mode for Rapid Drafting (Optional)
When actively writing and revising text, enable draft mode for images in `main.tex`:
```latex
\usepackage[draft]{graphicx}   % Replaces images with placeholder boxes
```
Switch back to `\usepackage{graphicx}` for the final camera-ready compilation.

---

## 3. Remove Obsolete & Unused Packages
In `main.tex`:
- Remove `\usepackage{subfigure}` (obsolete package causing deprecation warnings; no subfigures are utilized).
- Remove `\usepackage{epsfig}` (redundant legacy wrapper over `graphicx`).
- Comment out or remove unused layout packages (`mdframed`, `amsxtra`) to reduce TeX macro expansion overhead.

---

## 4. Enable PDF Stream Compression
Add the following compression parameters near the top of `main.tex` (directly following `\documentclass`):
```latex
\pdfcompresslevel=9
\pdfobjcompresslevel=2
```
This forces pdflatex to compress PDF object streams, shortening output I/O time.

---

## 5. Remove Orphan Files from Upload
- `abbr.tex` is currently not included via `\input` and can either be integrated into an acronym environment or removed from the Overleaf project bundle.
- Remove unreferenced figures from `figures/` if not planned for inclusion (`JU-logo.png`, `noise_combined_heatmap.png`, `sparsity_rep_comparison.png`, `fig2_top25_symptom_prevalence.png`, `fig3_symptoms_per_patient_distribution.png`).

---

## 6. Fast Single-Chapter Compilation (`\includeonly`)
For targeted work on a single chapter, convert `\input{Chap...}` to `\include{Chap...}` and use `\includeonly`:
```latex
% In main.tex preamble:
% \includeonly{Chap5/Chapter5}
```
This enables compiling only the active chapter in under 5 seconds while preserving global page numbers and cross-references.
