# AFFECTOR: Emotional Geometry Transfer through Transformer Activation Steering

Uses the official ACM acmart class with manuscript, screen, and nonacm options. This is an independently distributed report, not an assertion of ACM publication or acceptance. No venue, DOI, or copyright transfer is invented.

## Build

With a current TeX Live/MacTeX installation including acmart:

    latexmk -pdf AFFECTOR-Technical-Report-ACM.tex

Or Tectonic 0.17.0:

    tectonic --keep-logs --keep-intermediates AFFECTOR-Technical-Report-ACM.tex

The bibliography uses ACM-Reference-Format.bst. The generated .bbl is included; references.bib remains editable. All packages are standard TeX distribution packages. No shell escape is needed. The manuscript layout is single-column; change manuscript to sigconf for a venue requiring two columns, then recheck layout.

Includes the source Markdown snapshot and citation verification notes. Equations, tables, and citations are native LaTeX. AI disclosure remains above the suggested citation at the end.

Original report: https://github.com/FL2744/AFFECTOR/blob/main/docs/TECHNICAL_REPORT.md
