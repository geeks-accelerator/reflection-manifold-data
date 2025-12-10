# arXiv Submission

## Recommended Category

**cs.CL** - Computation and Language

Secondary: cs.AI (Artificial Intelligence), cs.LG (Machine Learning)

## Files Included

| File | Description |
|------|-------------|
| `reflective-manifold.tex` | LaTeX source (64 KB) |
| `reflective-manifold.bbl` | Pre-compiled bibliography (4.6 KB) |
| `figures/` | All 7 figures (3.7 MB total) |
| `arxiv-submission.tar.gz` | Ready-to-submit tarball (3.2 MB) |

## Compilation Instructions

The `.bbl` file is included, so bibtex is **not required**. To compile:

```bash
pdflatex reflective-manifold.tex
pdflatex reflective-manifold.tex  # Run twice for references
```

Or use the pre-compiled PDF in [paper/](../paper/).

## Tarball Verification

Verify tarball contents match source files:

```bash
tar -tzf arxiv-submission.tar.gz | sort
```

Expected contents:
```
./
./figures/
./figures/figure1_design_schematic.png
./figures/figure2_two_panel_timeseries.png
./figures/figure3_divergence_fan.png
./figures/figure4_correlation_grid.png
./figures/figure5_umap_twin.png
./figures/figure6_variance_bars.png
./figures/figure7_bootstrap_table.png
./reflective-manifold.bbl
./reflective-manifold.tex
```

## Pre-Submission Checklist

- [ ] Verify tarball compiles without errors on arXiv's TeX Live
- [ ] Check all figure references resolve
- [ ] Confirm author names and affiliations
- [ ] Review abstract for clarity
- [ ] Select appropriate categories (cs.CL primary)

## Submission Process

1. Go to https://arxiv.org/submit
2. Upload `arxiv-submission.tar.gz`
3. Select category: **cs.CL** (Computation and Language)
4. Add metadata (title, abstract, authors)
5. Submit for moderation

## Figures

| Figure | Description |
|--------|-------------|
| figure1_design_schematic.png | Experimental design overview |
| figure2_two_panel_timeseries.png | Variance evolution across loops |
| figure3_divergence_fan.png | Trajectory divergence patterns |
| figure4_correlation_grid.png | Cross-model correlation matrix |
| figure5_umap_twin.png | UMAP manifold visualization |
| figure6_variance_bars.png | Variance decomposition by factor |
| figure7_bootstrap_table.png | Bootstrap confidence intervals |

---

*See [paper/](../paper/) for the compiled PDF.*
