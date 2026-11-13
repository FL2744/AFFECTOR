# AFFECTOR: Coordinate-Guided Text Rewriting through Transformer Activation Interventions

Uses the official ACM acmart class with manuscript, screen, and nonacm options. This is an independently distributed report, not an assertion of ACM publication or acceptance. No venue, DOI, or copyright transfer is invented.

## Build

With a current TeX Live/MacTeX installation including acmart:

    latexmk -pdf AFFECTOR-ACM.tex

Or Tectonic 0.17.0:

    tectonic --keep-logs --keep-intermediates AFFECTOR-ACM.tex

The bibliography uses ACM-Reference-Format.bst. The generated .bbl is included; references.bib remains editable. All packages are standard TeX distribution packages. No shell escape is needed. The manuscript layout is single-column; change manuscript to sigconf for a venue requiring two columns, then recheck layout.

Includes the source Markdown snapshot and citation verification notes. Equations, tables, and citations are native LaTeX. AI disclosure appears immediately before the references.

Original report: https://huggingface.co/will-taggart/AFFECTOR/blob/main/docs/TECHNICAL_PAPER.md

Build verified with Tectonic 0.17.0 and bundled acmart v1.83 (2022/02/19). All 12 bibliography entries resolve.
