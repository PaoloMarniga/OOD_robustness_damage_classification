# OOD Robustness in Satellite-Based Disaster Damage Classification

## Overview

This repository contains the code used for the Master's thesis:

**"Evaluating Representation-Level and Optimization-Level Robustness Strategies for Location-Based Out-of-Distribution Generalization in Satellite Disaster Damage Classification"**

The project investigates how different robustness strategies affect model performance under strict location-based out-of-distribution (OOD) evaluation using the xView2 disaster damage assessment dataset.

The repository includes:

* Dataset exploration and preprocessing
* Benchmark split analysis
* Location-based OOD split construction
* Baseline ResNet50 implementation
* Supervised Contrastive Learning (SupCon)
* DRO-inspired optimization
* Benchmark and OOD evaluation
* Robustness analysis

---

# Research Objective

The objective of this thesis is to evaluate how different robustness strategies affect satellite-based disaster damage classification under realistic deployment conditions.

The study focuses on three main objectives:

* Quantifying the performance gap between standard benchmark evaluation and deployment-oriented OOD evaluation.
* Developing and validating a strict location-based OOD evaluation protocol.
* Comparing representation-level and optimization-level robustness strategies under identical experimental conditions.

Rather than proposing a novel architecture, the thesis emphasizes controlled empirical evaluation of robustness under distribution shift.

---

# Repository Structure

```text
OOD_robustness_damage_classification/

models/
│
├── classification_baseline.py
├── OOD_classification_baseline_unweighted.py
├── OOD_dro_classifier.py
└── OOD_SupCon.py

A. pre processing/
│
├── 1. exploratory_data_analysis.ipynb
├── 2. baseline_split_preprocessing.ipynb
└── 3. out-of-distribution_split_preprocessing.ipynb

B. deployment/
│
├── 4. original_baseline_model.ipynb
├── 5. OOD_baseline_unweighted.ipynb
├── 6. OOD_supcon.ipynb
└── 7. OOD_dro.ipynb

Original_README.md
Original_xView2_License copy.md
requirements.txt
README.md
```

---

# Dataset

The experiments use the xView2 / xBD disaster damage assessment dataset.

Damage categories:

1. No Damage
2. Minor Damage
3. Major Damage
4. Destroyed

The dataset contains multiple disaster events:

* Hurricane Harvey
* Hurricane Matthew
* Hurricane Florence
* Hurricane Michael
* Mexico Earthquake
* Palu Tsunami
* Santa Rosa Wildfire
* SoCal Fire
* Midwest Flooding
* Guatemala Volcano

---

# Experimental Design

Unlike the original xView2 competition pipeline, this thesis isolates the damage classification task from localization.

The workflow is:

1. Read ground truth building polygons from label JSON files.
2. Construct a building-level metadata table.
3. Extract pre-event and post-event building crops.
4. Stack pre-event and post-event crops into 6-channel tensors.
5. Train ResNet50-based damage classification models.
6. Evaluate performance under benchmark and location-based OOD protocols.

This design removes localization as a confounding factor and allows controlled evaluation of classification robustness.

---

# Evaluation Protocols

Two evaluation settings are considered.

## Benchmark Evaluation

The original xView2 benchmark split is used as a reference condition.

This setting reflects the standard evaluation protocol commonly used in the literature and competition environment.

## Location-Based OOD Evaluation

A custom location-disjoint split is constructed:

```text
OOD_train
OOD_test
OOD_hold
```

Disaster locations are separated across training, validation, and final evaluation partitions.

This protocol is designed to better approximate deployment conditions in which a model encounters disaster environments that were not observed during training.

---

# Required Data

Download the xView2 dataset and place the following folders on desktop:

```text
train/
test/
hold/
```

These folders contain:

* Satellite imagery
* Label JSON files

---

# Required Software

* Python 3.10
* Miniconda (recommended)
* Jupyter Notebook or VS Code
* Git

---

# Environment Setup

Create the environment:

```bash
conda create -n thesis_xview python=3.10

conda activate thesis_xview

python --version
which python

python -m pip install --upgrade pip setuptools wheel

python -m pip install -r requirements.txt

python -m ipykernel install \
    --user \
    --name thesis_xview \
    --display-name "Python thesis_xview"
```

---

# Kernel Setup

Restart VS Code or Jupyter and select:

```text
Python thesis_xview
```

from:

```text
Kernel → Change Kernel → Python thesis_xview
```

---

# Workflow

Recommended execution order.

## Stage 1: Exploratory Analysis

Run:

```text
1. exploratory_data_analysis.ipynb
```

Purpose:

* Explore dataset composition
* Analyze disaster distribution
* Analyze class imbalance
* Validate crop generation
* Produce summary tables

---

## Stage 2: Dataset Preparation

Run:

```text
2. baseline_split_preprocessing.ipynb
3. out-of-distribution_split_preprocessing.ipynb
```

Purpose:

* Create benchmark split metadata
* Create location-based OOD split metadata
* Validate split composition
* Verify absence of overlap

---

## Stage 3: Model Evaluation

Run:

```text
4. original_baseline_model.ipynb
```

Purpose:

* Train and evaluate the benchmark baseline using the original xView2 split.

---

## Stage 4: OOD Evaluation

Run:

```text
5. OOD_baseline_unweighted.ipynb
6. OOD_supcon.ipynb
7. OOD_dro.ipynb
```

Purpose:

* Evaluate the OOD baseline
* Evaluate supervised contrastive learning
* Evaluate DRO-inspired optimization
* Generate robustness metrics

---

# Models

The repository evaluates three approaches.

## Baseline

Standard ResNet50 damage classifier trained using cross-entropy loss.

Implementation:

```text
models/classification_baseline.py
```

## SupCon

Supervised contrastive representation learning combined with a ResNet50 backbone.

Implementation:

```text
models/OOD_SupCon.py
```

## DRO-inspired

Distributionally robust optimization inspired objective emphasizing high-risk environments.

Implementation:

```text
models/OOD_dro_classifier.py
```

---

# Main Research Questions

1. How large is the performance gap between benchmark evaluation and location-based OOD evaluation?

2. Can a location-based OOD protocol better approximate deployment conditions?

3. How do representation-level robustness strategies affect OOD performance?

4. How do optimization-level robustness strategies affect OOD performance?

5. How do representation-level and optimization-level robustness strategies compare under location-based OOD evaluation?

---

# Reproducibility

The experiments use the following random seeds:

```python
[42, 123, 999, 2024, 2025]
```

All models use:

* The same ResNet50 backbone
* The same crop generation pipeline
* The same input representation
* The same evaluation protocol
* Comparable computational budgets

This allows differences in performance to be more directly attributed to the robustness strategy under evaluation.

### Important note

The location-based OOD split is generated using Gurobi via `gurobipy`. A valid Gurobi license is required to rerun the optimization step. The trained model experiments use the generated OOD split files and do not require rerunning the optimizer once those files have been created.

---

# Acknowledgements

Parts of the implementation are adapted from the official xView2 baseline repository.

The original documentation and license are included in:

```text
Original_README.md
Original_xView2_License copy.md
```

All preprocessing, OOD split construction, training adaptations, evaluation procedures, and robustness experiments were developed for this thesis.
