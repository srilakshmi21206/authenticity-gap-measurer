# Authenticity Gap Measurer

**Authenticity Gap Measurer** is an AI-driven framework for evaluating whether synthetic image datasets represent the rare, difficult, and operationally important cases found in real-world data. Instead of relying only on aggregate distribution comparisons, it identifies unusual real images, measures their coverage by synthetic samples, explains characteristics associated with missing coverage, and searches for improved synthetic-data mixtures under a fixed image-generation budget.

Developed for **Problem Statement #6272FBE4 — AI & IoT, आविष्KAR FET Hackathon**.

---

## Overview

A synthetic dataset can appear similar to real data in aggregate while still failing to represent rare or challenging examples. Authenticity Gap Measurer focuses on this potential blind spot by answering four questions:

1. Which real images are unusual within their class?
2. Which unusual images lack a sufficiently similar synthetic counterpart?
3. What visual characteristics are associated with the missing cases?
4. Can the synthetic-data generation strategy be improved without increasing the image-generation budget?

The framework's primary metric, the **Authenticity Gap Score**, measures the proportion of real edge cases that are not sufficiently covered by the synthetic dataset.

## Key Features

- Detects rare or atypical real-world samples within each class.
- Measures synthetic coverage of difficult real samples using DINOv2 embeddings.
- Produces a ranked gallery of uncovered real edge cases.
- Summarizes visual traits associated with missing coverage.
- Tests robustness across edge-case fractions and coverage thresholds.
- Validates findings using a classifier trained on synthetic data and evaluated on held-out real images.
- Uses a genetic algorithm to optimize the mixture of synthetic-data generators under a fixed image budget.
- Provides an interactive Streamlit dashboard, including support for user-uploaded real and synthetic image datasets.

## How It Works

### 1. Feature extraction

Each real and synthetic image is mapped to a semantic feature embedding using the DINOv2 vision transformer model:

```text
facebook/dinov2-small
```

These embeddings provide a shared feature space for comparing visual similarity beyond raw pixel values.

```bash
python extract_features.py
```

### 2. Edge-case detection

For each real image, the system estimates how unusual it is relative to other real images in the same class. By default, the most unusual **10%** of real images in each class are designated as edge cases.

```bash
python find_edge_cases.py
```

In the evaluated dataset, this step identified approximately **876 edge cases among more than 8,760 real images**.

### 3. Synthetic coverage analysis

For each real image, the framework finds the nearest synthetic image from the same class using cosine distance in DINOv2 embedding space. For a real image embedding \(r\), the synthetic embeddings for class \(c\), denoted by \(S_c\), and the class \(c\):

\[
d(r, S_c) = \min_{s \in S_c} \left(1 - \cos(r, s)\right)
\]

A class-specific coverage threshold is calculated from normal (non-edge) real examples. By default, it is the **95th percentile** of their nearest-synthetic distances. An edge case is considered uncovered when:

\[
d(r, S_c) > T_c
\]

where \(T_c\) is the threshold for class \(c\).

```bash
python coverage_analysis.py
```

## Authenticity Gap Score

The score is the percentage of real edge cases that are not sufficiently represented by the synthetic dataset:

\[
\text{Authenticity Gap Score} =
\frac{\text{Number of uncovered edge cases}}
{\text{Total number of edge cases}}
\times 100
\]

A higher score indicates that a larger proportion of difficult real-world samples lack adequate synthetic coverage. Because the default threshold is the 95th percentile of normal examples, an uncovered rate of approximately **5%** is expected by design. Interpret results in light of this chance baseline; the excess uncovered rate is the more meaningful signal.

## Missing-Case Explanations

The framework compares automatically inferred visual traits among uncovered edge cases, covered edge cases, and ordinary real images. Heuristic traits include:

- Brightness and contrast
- Sharpness and edge density
- Partial object visibility
- Small objects or leaves in the frame
- Cropped framing
- Low-detail or blurry appearance

The analysis can reveal patterns such as dark, blurry, tightly cropped, partially visible, or small-in-frame objects being over-represented among uncovered samples. These tags are automatically inferred and should not be treated as human-verified annotations.

## Validation and Optimization

### Robustness analysis

The score can be recalculated using different edge-case fractions, coverage percentiles, and per-class or global thresholds. This helps assess whether a finding is stable rather than an artifact of a single threshold choice.

```bash
python robustness_analysis.py features report_v2
```

For the mild-augmentation dataset:

```bash
python robustness_analysis.py features_v1 report_v1
```

### Before-and-after comparison

Compare synthetic-data strategies—for example, adding random crops or changing augmentation severity—and review per-class changes in edge-case coverage.

```bash
python compare_before_after.py
```

### Classifier validation

As an independent validation signal, a classifier is trained using synthetic data only and evaluated on held-out real images. The real test set is grouped into covered edge cases, uncovered edge cases, and normal examples. If the coverage metric is informative, accuracy is generally expected to be lower on uncovered edge cases.

```bash
python accuracy_check.py
python confusion_matrix.py
```

### Evolutionary synthetic-data optimization

A genetic algorithm searches for an effective mixture of synthetic-data generators while keeping the total image budget fixed. The evaluated generator types are:

- No crop
- Gentle crop
- Aggressive crop

The algorithm optimizes the Authenticity Gap Score and a coverage-related objective. **It does not directly optimize classifier accuracy**; accuracy is reported separately as an independent validation signal.

```bash
python Evolution_engine.py
python make_evolution_bundle.py
```

The dashboard also includes an unsupervised optimizer at `streamlit_app/evolution_unsupervised.py`. It does not use class labels during optimization; labels are used only for subsequent validation of classification accuracy on held-out real images.

## Results

Two synthetic datasets were evaluated while keeping the real images, edge-case flags, and coverage thresholds fixed. The synthetic generation pipeline was the changed variable.

| Version | Synthetic augmentation strategy | Authenticity Gap Score |
|---|---|---:|
| v1 | Mild augmentation | **33.0%** |
| v2 | Strong augmentation with heavy crops, dropout, elastic distortion, and grid distortion | **45.4%** |

The stronger augmentation pipeline produced a larger gap across all classes, indicating that the score responds to changes in the synthetic-data pipeline. At the strictest coverage setting, both versions were approximately **20–22%**, suggesting that the hardest real-world edge cases remained uncovered regardless of augmentation strength. Approximately 5% of the measured gap is expected from the default threshold by design.

In five independent runs over 60 generations, evolutionary optimization improved the held-out gap score under the same total synthetic-image budget:

| Method | Held-out Authenticity Gap Score |
|---|---:|
| Best individual synthetic generator | Approximately **49%** |
| Evolved generator mixture | Approximately **41%** |

Saved evolutionary outputs include `report_v1/evolution_summary.csv` and `report_v1/evolution_convergence.png`.

## Interactive Dashboard

Launch the Streamlit application with:

```bash
streamlit run app.py
```

The dashboard includes:

| Tab | Purpose |
|---|---|
| Overview | Headline score, class-level results, and key observations |
| Why cases go missing | Visual traits over-represented among uncovered edge cases |
| Missing edge cases | Filterable gallery of uncovered real images |
| Fix testing | Before-and-after comparison of synthetic-data strategies |
| Accuracy | Classifier performance on covered and uncovered real cases |
| Robustness | Metric results across thresholds and edge-case fractions |
| Auto evolve | Unsupervised genetic optimization under a fixed image budget |
| Try your own data | Evaluation of uploaded real and synthetic `.zip` image datasets |
| About | Methodology, assumptions, and limitations |

The **Auto evolve** tab reports individual and evolved-mixture gap scores, classification accuracy after optimization, selected generator mixtures by discovered group, and progress across generations.

## Installation and Reproduction

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Run the analysis pipeline:

```bash
python extract_features.py
python find_edge_cases.py
python coverage_analysis.py
python robustness_analysis.py features report_v2
python compare_before_after.py
python accuracy_check.py
python confusion_matrix.py
python Evolution_engine.py
python make_evolution_bundle.py
streamlit run app.py
```

For robustness analysis on the mild-augmentation dataset, use:

```bash
python robustness_analysis.py features_v1 report_v1
```

The project expects real and synthetic images to be organized by class. Dataset images, feature embeddings, and generated reports are stored in project directories described below; large data and derived files are excluded from the repository. Refer to the scripts and configuration in the repository for the exact paths used by a particular run.

## Repository Structure

```text
Authenticity_Gap_Project/
├── real/                            # Real images organized by class (not committed)
├── synthetic*/                      # Synthetic image sets (not committed)
├── features*/                       # DINOv2 feature embeddings (not committed)
├── report/                          # Temporary outputs; overwritten on each run
├── report_v1/                       # Mild-augmentation results and GA outputs
├── report_v2/                       # Strong-augmentation results
├── streamlit_app/
│   ├── evolution_unsupervised.py    # Label-free GA engine used by the dashboard
│   └── evolution_bundle.npz          # PCA-compressed data for live dashboard optimization
├── app.py                           # Streamlit dashboard
├── extract_features.py              # DINOv2 feature extraction
├── find_edge_cases.py               # Edge-case identification
├── coverage_analysis.py             # Coverage analysis and gap score
├── robustness_analysis.py           # Threshold and edge-fraction robustness tests
├── compare_before_after.py          # Synthetic-strategy comparison
├── accuracy_check.py                # Synthetic-only classifier evaluation on real images
├── confusion_matrix.py              # Confusion-matrix generation
├── Evolution_engine.py              # Label-aware genetic optimization engine
├── make_evolution_bundle.py         # Dashboard bundle generation
├── generate_synthetic.py            # Mild augmentation pipeline
├── generate_synthetic_v2.py         # Strong augmentation pipeline
├── generate_synthetic_v3.py         # Additional augmentation pipeline
├── label_and_baseline.py            # Labeling and baseline-model utilities
├── requirements.txt                 # Python dependencies
├── .gitignore
└── README.md
```

## Limitations

- The evaluated synthetic images are augmented versions of real images, not independently generated samples. The measured coverage therefore reflects augmentation diversity as well as synthetic realism.
- A real image may be compared with an augmented version of itself, which can make the reported gap score conservative.
- The default 95th-percentile threshold implies an expected chance-level uncovered rate of approximately 5%.
- Classes with few edge cases can produce unstable estimates; small-class scores should be interpreted cautiously.
- The evolutionary engine optimizes the gap score and a coverage objective, not downstream classification accuracy.
- The dashboard's unsupervised optimizer does not use labels during optimization; labels are used only for later evaluation.
- Visual tags such as brightness, sharpness, and partial-object presence are heuristic and have not been human-verified.
- DINOv2 embedding distance is a proxy for semantic similarity and may not capture every domain-specific property relevant to deployment.

## Future Work

- Evaluate independently generated images from diffusion or GAN-based pipelines.
- Add domain-expert annotations for edge-case categories.
- Incorporate uncertainty estimates into coverage thresholds.
- Compare additional embedding models and domain-specific feature extractors.
- Support object-level and region-level coverage analysis.
- Extend optimization to image-generation parameters such as illumination, blur, viewpoint, scale, and background complexity.
- Add active-learning support to prioritize collection of real samples from uncovered regions.
- Study the relationship between the Authenticity Gap Score and deployment error across more datasets and domains.

## Conclusion

Authenticity Gap Measurer provides an actionable alternative to aggregate real-versus-synthetic similarity checks. It identifies difficult real-world cases that synthetic data does not adequately represent, investigates why coverage is missing, tests proposed fixes, and searches for an improved synthetic-data mixture without increasing the total image budget.

```text
Detect missing real-world cases
→ Explain why they are missing
→ Test candidate fixes
→ Evolve a better synthetic-data mixture
```
