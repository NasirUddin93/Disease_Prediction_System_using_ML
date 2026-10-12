"""
Script to modularize Paper_Writing/main.tex into chapter/section files and optimize images for Overleaf
"""
import os
from pathlib import Path
from PIL import Image

def split_latex():
    base_dir = Path("Paper_Writing")
    sections_dir = base_dir / "sections"
    sections_dir.mkdir(exist_ok=True)

    main_tex_path = base_dir / "main.tex"
    with open(main_tex_path, "r", encoding="utf-8") as f:
        content = f.read()

    sec_markers = [
        ("% ==============================================================================\n% SECTION 1: INTRODUCTION", "01_introduction.tex"),
        ("% ==============================================================================\n% SECTION 2: RELATED WORK", "02_related_work.tex"),
        ("% ==============================================================================\n% SECTION 3: DATASET FORENSIC ANALYSIS", "03_dataset_forensics.tex"),
        ("% ==============================================================================\n% SECTION 4: METHODOLOGY & EXPERIMENTAL DESIGN", "04_methodology.tex"),
        ("% ==============================================================================\n% SECTION 5: EXPERIMENTAL HYPOTHESES", "05_hypotheses.tex"),
        ("% ==============================================================================\n% SECTION 6: EMPIRICAL BENCHMARK RESULTS & PROTOCOL COMPARISON", "06_benchmark_results.tex"),
        ("% ==============================================================================\n% SECTION 7: FEATURE SELECTION & DIMENSIONALITY REDUCTION", "07_feature_selection.tex"),
        ("% ==============================================================================\n% SECTION 8: CLINICAL SYMPTOM SPARSITY & COLD-START ROBUSTNESS", "08_symptom_sparsity.tex"),
        ("% ==============================================================================\n% SECTION 9: DISCUSSION & ETHICAL CONSIDERATIONS", "09_discussion.tex"),
        ("% ==============================================================================\n% SECTION 10: CONCLUSION", "10_conclusion.tex"),
        ("% ==============================================================================\n% REFERENCES", None)
    ]

    indices = []
    for marker, filename in sec_markers:
        pos = content.find(marker)
        if pos == -1:
            print("Warning: marker not found:", repr(marker[:40]))
        indices.append((pos, marker, filename))

    indices.sort(key=lambda x: x[0])

    # Preamble is before the first section
    preamble = content[:indices[0][0]].strip()

    # Extract each section
    for i in range(len(indices) - 1):
        start_pos = indices[i][0]
        end_pos = indices[i+1][0]
        fname = indices[i][2]
        sec_text = content[start_pos:end_pos].strip()
        out_file = sections_dir / fname
        with open(out_file, "w", encoding="utf-8") as out:
            out.write(sec_text + "\n")
        print(f"Wrote {fname} ({len(sec_text)} chars)")

    # Comment out microtype in preamble to prevent Overleaf free-tier compile timeout
    preamble_clean = preamble.replace(
        r"\usepackage{microtype}",
        r"% \usepackage{microtype} % NOTE: Commented out to eliminate Overleaf free-tier compile timeout"
    )

    # Build slim modular main.tex
    modular_main = preamble_clean + "\n\n" + r"""% ==============================================================================
% MODULAR CHAPTER & SECTION INCLUDES
% ==============================================================================
\input{sections/01_introduction}
\input{sections/02_related_work}
\input{sections/03_dataset_forensics}
\input{sections/04_methodology}
\input{sections/05_hypotheses}
\input{sections/06_benchmark_results}
\input{sections/07_feature_selection}
\input{sections/08_symptom_sparsity}
\input{sections/09_discussion}
\input{sections/10_conclusion}

% ==============================================================================
% REFERENCES
% ==============================================================================
\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
"""

    with open(main_tex_path, "w", encoding="utf-8") as f:
        f.write(modular_main.strip() + "\n")

    print("Successfully converted main.tex to modular architecture!")

    # Optimize figure images to prevent memory/time-out during PDF generation
    fig_dir = base_dir / "figures"
    for p in fig_dir.glob("*.png"):
        orig_size = os.path.getsize(p)
        try:
            im = Image.open(p)
            im.save(p, optimize=True)
            new_size = os.path.getsize(p)
            print(f"Optimized {p.name}: {orig_size//1024}KB -> {new_size//1024}KB")
        except Exception as e:
            print(f"Error optimizing {p.name}: {e}")

if __name__ == "__main__":
    split_latex()
