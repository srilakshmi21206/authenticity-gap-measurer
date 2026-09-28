# Authenticity Gap Measurer

Quantifies **specifically which real-world edge cases** synthetic training
data fails to cover — not a generic real-vs-synthetic distribution
comparison.

## Problem

Synthetic data is usually validated by comparing overall distributions
(real vs. synthetic feature statistics). That hides *which* real images the
synthetic data actually fails to represent. This project finds the unusual,
hard, real-world images (edge cases) in each class and measures how many of
them have no close match in the synthetic set.

## Method

1. **Feature extraction** (`extract_features.py`) — embed all real and
   synthetic images with DINOv2 (`facebook/dinov2-small`).
2. **Edge case detection** (`find_edge_cases.py`) — score every real image by
   how unusual it is within its class; flag the top 10% per class as edge
   cases (876 of 8,760 images).
3. **Coverage analysis** (`coverage_analysis.py`) — for each real image, find
   the cosine distance to its nearest synthetic image of the same class. A
   per-class threshold is set at the 95th percentile of that distance among
   *normal* (non-edge) images. An edge case is **uncovered** if its distance
   exceeds that threshold.
4. **Authenticity Gap Score** = % of edge cases that are uncovered.
5. **Robustness analysis** (`robustness_analysis.py`) — recomputes the score
   across different percentile thresholds (90th/95th/99th) and edge
   fractions (top 5–20%), and reports the excess over the chance level
   (since ~5% of normal images exceed the 95th-percentile cutoff by design).

## Results

Two synthetic sets were generated with different augmentation strength
(`generate_synthetic.py` for v1, `generate_synthetic_v2.py` for v2, using
Albumentations). The real images, edge-case flags (876 total), and
thresholds were identical across both runs — only the synthetic data
differs.

| Version | Synthetic augmentation | Gap score (95th pct, top 10%) | Uncovered |
|---|---|---|---|
| v1 | mild | **33.0** | 289 / 876 |
| v2 | strong (heavy crops, dropout, elastic/grid distortion) | **45.4** | 398 / 876 |

**Robustness:** v2 scores higher than v1 in 11 of 12 threshold settings
tested. At the strictest setting (99th percentile) both versions converge
(~20–22), meaning the hardest edge cases are uncovered under either
augmentation strategy — the gap between versions is driven by moderately
unusual cases. About 5% of any score is chance by design; excess over
chance at the base setting is 28.0 (v1) vs. 40.4 (v2).

**Per-class gap score, v1 → v2:**

| Class | v1 | v2 |
|---|---|---|
| Potato Early blight | 35.0% | 57.0% |
| Potato Late blight | 50.0% | 62.0% |
| Potato healthy* | 46.7% | 93.3% |
| Tomato Bacterial spot | 36.8% | 47.2% |
| Tomato Early blight | 45.0% | 59.0% |
| Tomato Late blight | 15.8% | 30.0% |
| Tomato healthy | 27.7% | 30.8% |

\* Only 15 edge cases in this class — too few to draw a reliable conclusion.

Full outputs, including the ranked list of uncovered images and a gallery of
the 24 worst-covered edge cases, are in [`report_v1/`](report_v1) and
[`report_v2/`](report_v2).

## Key finding

Coverage depends heavily on how the synthetic data is generated: stronger
augmentation widened the gap in every class. This shows the score is a
useful, sensitive diagnostic — but also that "coverage" as measured here
partly reflects augmentation strategy, not only underlying data realism,
since each synthetic image is an augmented copy of a real image.

## Demo

Run `python app.py` for an interactive Gradio dashboard: the overall
Authenticity Gap Score, a per-class bar chart, and a browsable gallery of
the real edge-case images the synthetic data fails to cover (filterable by
class, with distance-to-nearest-synthetic scores). It reads from `report/`,
so re-run the pipeline for the version you want to inspect before launching.

## Reproduce

```bash
python extract_features.py
python find_edge_cases.py
python coverage_analysis.py
python robustness_analysis.py features report_v2   # or features_v1 report_v1
python app.py                                       # interactive dashboard
```

## Repo structure


```
Authenticity_Gap_Project/
├── real/                      # real images, by class (not pushed — see .gitignore)
├── synthetic/                 # v1 synthetic output (not pushed)
├── synthetic_v1/              # v1 synthetic set, renamed (not pushed)
├── synthetic_v2/               # v2 (strong augmentation) synthetic output (not pushed)
├── features/                  # DINOv2 embeddings, v2 run (not pushed)
├── features_v1/                # DINOv2 embeddings, v1 run (not pushed)
├── report/                    # scratch output — overwritten on every run
├── report_v1/                 # saved results, v1 (mild augmentation)
├── report_v2/                 # saved results, v2 (strong augmentation)
│   ├── class_report.csv
│   ├── missing_edge_cases.csv
│   ├── per_class_gap.csv / .png
│   ├── robustness_grid.csv
│   ├── robustness_excess.csv
│   ├── robustness_heatmap.png
│   └── worst_missing_gallery.png
├── app.py                     # Gradio dashboard for exploring results
├── extract_features.py        # DINOv2 embeddings for real + synthetic images
├── find_edge_cases.py         # flags top-10%-per-class unusual real images
├── coverage_analysis.py       # computes the Authenticity Gap Score
├── robustness_analysis.py     # score across thresholds/fractions + per-class breakdown
├── generate_synthetic.py      # v1 augmentation pipeline
├── generate_synthetic_v2.py   # v2 (stronger) augmentation pipeline
├── label_and_baseline.py      # labeling / baseline model utilities
├── requirements.txt
├── .gitignore
└── README.md
```

## Limitations

- Synthetic images are augmentations of real images, not independently
  generated — coverage reflects augmentation diversity as much as data
  quality.
- The 95th-percentile threshold has a built-in ~5% chance-level uncovered
  rate.
- The `Potato___healthy` class has too few edge cases (15) for its gap score
  to be reliable.