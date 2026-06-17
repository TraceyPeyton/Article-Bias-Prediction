# arXiv Manuscript Package

This folder contains a LaTeX draft derived from `Bias Classifier Report.docx`
and prepared for arXiv-style submission.

## Files

- `main.tex`: top-level LaTeX manuscript.
- `references.bib`: BibTeX references.
- `figures/report_figure_1.png`: figure extracted from the report DOCX.

## Build

From this folder:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The resulting `main.pdf` is the local build artifact. arXiv generally prefers
the source package rather than only a PDF, so upload `main.tex`,
`references.bib`, and `figures/report_figure_1.png`.

## Notes Before Submission

- Confirm all authors approve public posting.
- Confirm the intended arXiv category, likely `cs.CL` or `cs.LG`.
- Review the Hugging Face `matous-volf/political-leaning-politics` license note:
  the model card lists `cc-by-nc-4.0`.
- Remove build byproducts such as `.aux`, `.bbl`, `.blg`, `.log`, `.out`, and
  `.pdf` from the final source upload unless arXiv asks for them.
