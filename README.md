# Week 3 – Iris Flower Clustering

## Overview

This project demonstrates unsupervised machine learning using the Scikit-learn Iris dataset, which contains 150 samples and 4 features.

## Workflow

- Standardized the features with `StandardScaler`.
- Used the Elbow method to select `K=3`.
- Applied K-Means clustering with `K=3`.
- Obtained cluster counts of 53, 50, and 47.
- Achieved an Adjusted Rand Index (ARI) of `0.6201` when comparing clusters with the known Iris labels.
- Reduced the 4-feature data to 2 principal components with PCA. The two components retain `95.81%` of the variance: PC1 retains `72.96%` and PC2 retains `22.85%`.

## Prediction Example

For [5.1, 3.5, 1.4, 0.2], the saved pipeline assigns the sample to cluster `1`. The cluster comparison associated cluster `1` with **Setosa**.

## Saving and Loading

The K-Means model and scaler are saved and loaded with Joblib for reuse.

```python
import joblib

joblib.dump(kmeans, "iris_kmeans_model.pkl")
joblib.dump(scaler, "iris_scaler.pkl")

kmeans = joblib.load("iris_kmeans_model.pkl")
scaler = joblib.load("iris_scaler.pkl")
```

## Installation

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

## Project Files

- `README.md`
- `week3_iris_flower_clustering.ipynb` - notebook (not uploaded in this session)
- `iris_kmeans_model.pkl` - saved K-Means model (not uploaded in this session)
- `iris_scaler.pkl` - saved StandardScaler (not uploaded in this session)

The notebook and model artifacts are intentionally not uploaded because the provided files were unavailable in the cloud workspace.
