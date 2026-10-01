# Authenticity Gap Measurer

Quantifies **specifically which real-world edge cases** synthetic training
data fails to cover — then **evolves a fix** for it, rather than stopping at
a generic real-vs-synthetic distribution comparison.

**Problem statement #6272FBE4** · AI & IoT · आविष्KAR FET Hackathon

## Problem

Synthetic data is usually validated by comparing overall distributions
(real vs. synthetic feature statistics). That hides *which* real images the
synthetic data actually fails to represent. This project finds the unusual,
hard, real-world images (edge cases) in each class, measures how many of
them have no close match in the synthetic set, explains *why* they're
missing, and then uses a genetic algorithm to automatically search for a
better mix of synthetic generators that closes the gap.

## Method

1. **Feature extraction** (`extract_features.py`) — embed every real and
   synthetic image with DINOv2 (`facebook/dinov2-small`).
2. **Edge case detection** (`find_edge_cases.py`) — score every real image
   by how unusual it is within its class; flag the rarest 10% per class as
   edge cases (876 of 8,760+ images).
3. **Coverage analysis** (`coverage_analysis.py`) — for each real image,
   find the cosine distance to its nearest synthetic image of the same
   class. A per-class threshold is set at the 95th percentile of that
   distance among *normal* (non-edge) images. An edge case is **uncovered**
   if its distance exceeds that threshold.
4. **Authenticity Gap Score** = % of edge cases that are uncovered.
5. **Explain** (tag analysis) — heuristic image tags (brightness, sharpness,
   edge density, partial/small leaf in frame...) are compared between
   missing edge cases and ordinary real images, to say *why* cases go
   missing, not just *that* they do.
6. **Validate**:
   - `robustness_analysis.py` — recomputes the score across different
     percentile thresholds and edge fractions, and reports the excess over
     the chance level (about 5% of normal images exceed a 95th-percentile
     cutoff by design).
   - `compare_before_after.py` — tests whether a specific fix (e.g. adding
     random crops) actually closes the gap, per class.
   - `accuracy_check.py` + `confusion_matrix.py` — trains a classifier on
     synthetic-only data and checks accuracy on held-out real images,
     split by covered vs. uncovered edge cases, as an independent check on
     whether the score predicts real failures.
7. **Fix it — Evolutionary engine** (`Evolution_engine.py`,
   `make_evolution_bundle.py`) — rather than only diagnosing the gap, a
   genetic algorithm searches for the best **mix** of synthetic generators
   (no crop / gentle crop / aggressive crop) per class, under the same
   total synthetic-image budget as the baseline, to minimize the held-out
   gap score. An **unsupervised** version
   (`streamlit_app/evolution_unsupervised.py`) repeats this without using
   any class labels during optimisation — labels are used only afterwards,
   to validate accuracy on held-out real images.

## Dashboard

`app.py` is a Streamlit dashboard with the following tabs:

- **Overview** — headline score, per-class breakdown, key findings
- **Why cases go missing** — which image traits are over-represented among
  missing edge cases
- **Missing edge cases** — a filterable, browsable gallery of the actual
  real images synthetic data fails to cover, with per-image stats
- **Fix testing** — before/after comparison of a proposed fix (e.g. adding
  crops), per class
- **Accuracy** — classifier accuracy trained on synthetic vs. real data,
  tested on held-out real images, split by covered/uncovered edge cases
- **Robustness** — the gap score recomputed across thresholds and edge
  fractions, to confirm it isn't an accident of one setting
- **Auto evolve** — runs the unsupervised genetic algorithm live in the
  browser: groups real images without labels, finds edge cases, and
  evolves the best mix of synthetic generators to minimize the gap score
  under a fixed budget. Reports held-out gap score and accuracy per method
  (each generator alone vs. the evolved mix), plus the evolved mixture per
  group.
- **Try your own data** — upload your own real/synthetic `.zip` sets and
  get your own Authenticity Gap Score computed live
- **About** — method and limitations

Run it with:
```bash
streamlit run app.py
```

## Results

Two synthetic sets were generated with different augmentation strength
(`generate_synthetic.py` for v1/mild, `generate_synthetic_v2.py` for
v2/strong). The real images, edge-case flags, and thresholds were
identical across runs — only the synthetic data differs.

| Version | Synthetic augmentation | Gap score (95th pct, top 10%) |
|---|---|---|
| v1 | mild | **33.0** |
| v2 | strong (heavy crops, dropout, elastic/grid distortion) | **45.4** |

Stronger augmentation widened the gap in every class — the score is
genuinely sensitive to how synthetic data is generated, not a fixed
number. At the strictest threshold setting, both versions converge
(~20–22), meaning the hardest edge cases are uncovered either way; the
difference between versions comes from moderately unusual cases. About 5%
of any score is chance by design.

**Evolutionary search result:** mixing three synthetic generators (no
crop / gentle crop / aggressive crop) per class, under the same total
image budget as the baseline, lowered the held-out gap score from
**~49 (best single generator) to ~41 (evolved mix)** over 60 generations,
consistently across 5 independent runs — see
`report_v1/evolution_summary.csv` and `evolution_convergence.png`. This
shows the gap can be *reduced*, not just measured, without generating any
additional synthetic images.

Full outputs are in [`report_v1/`](report_v1) and [`report_v2/`](report_v2),
including the ranked list of uncovered images, a gallery of the worst
cases, and the evolutionary engine's results.

## Reproduce

```bash
python extract_features.py
python find_edge_cases.py
python coverage_analysis.py
python robustness_analysis.py features report_v2   # or features_v1 report_v1
python compare_before_after.py
python accuracy_check.py
python confusion_matrix.py
python Evolution_engine.py              # label-aware GA (writes to report_v1/)
python make_evolution_bundle.py         # builds the bundle for the dashboard's Auto Evolve button
streamlit run app.py                    # interactive dashboard, including live Auto Evolve
```

## Repo structure
```
Authenticity_Gap_Project/
├── real/                       # real images, by class (not pushed — see .gitignore)
├── synthetic*/                 # synthetic sets, mild/strong/aggressive (not pushed)
├── features*/                  # DINOv2 embeddings per synthetic version (not pushed)
├── report/                     # scratch output — overwritten on every run
├── report_v1/                  # saved results, v1 (mild augmentation) + evolution engine results
├── report_v2/                  # saved results, v2 (strong augmentation)
├── streamlit_app/
│   ├── evolution_unsupervised.py   # label-free GA engine used by the dashboard's Auto Evolve tab
│   └── evolution_bundle.npz        # precomputed, PCA-compressed data for the live Auto Evolve button
├── app.py                      # Streamlit dashboard (Overview, Missing cases, Fix testing, Accuracy, Robustness, Auto evolve, Try your own data, About)
├── extract_features.py         # DINOv2 embeddings for real + synthetic images
├── find_edge_cases.py          # flags the rarest 10%-per-class real images
├── coverage_analysis.py        # computes the Authenticity Gap Score
├── robustness_analysis.py      # score across thresholds/fractions + per-class breakdown
├── compare_before_after.py     # tests whether a proposed fix closes the gap, per class
├── accuracy_check.py           # classifier accuracy: trained on synthetic, tested on real
├── confusion_matrix.py         # confusion matrix for the accuracy check
├── Evolution_engine.py         # label-aware genetic algorithm (mixes generators per class)
├── make_evolution_bundle.py    # builds streamlit_app/evolution_bundle.npz for the dashboard
├── generate_synthetic.py       # v1 (mild) augmentation pipeline
├── generate_synthetic_v2.py    # v2 (strong) augmentation pipeline
├── generate_synthetic_v3.py    # v3 augmentation pipeline
├── label_and_baseline.py       # labeling / baseline model utilities
├── requirements.txt
├── .gitignore
└── README.md
```

## Limitations

- Synthetic images are augmentations of real images, not an independently
  generated set — coverage reflects augmentation diversity as much as
  underlying data realism.
- A real image's nearest synthetic match can be its own augmented copy,
  which makes the headline score conservative.
- The 95th-percentile threshold has a built-in ~5% chance-level uncovered
  rate.
- Tiny classes (e.g. Potato healthy, 15 edge cases) are too small for a
  reliable gap score.
- The evolutionary engine's fitness is the gap score (plus a coverage
  term), computed without labels; classification accuracy is reported only
  afterwards, as independent validation — it is not what the GA optimizes.
- Heuristic image tags (brightness, sharpness, etc.) are not human-verified.

## Demo

Run `streamlit run app.py` for the full interactive dashboard, including
the live "Auto evolve" button.
