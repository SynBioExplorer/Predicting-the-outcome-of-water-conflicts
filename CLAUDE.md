# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Predicts the outcome of international transboundary water conflicts using ML classification models. Built as a single Jupyter notebook (`final_project_iron_hack.ipynb`) originally run on Google Colab.

Data source: Transboundary Freshwater Dispute Database (Oregon State University).

## Architecture

The notebook (269 cells) follows a linear pipeline:

1. **Data ingestion** (cells ~7-10): Loads four datasets from Google Drive Excel/shapefile sources
   - `df1`: Country-level context (UN AQUASTAT - political, socioeconomic, water resources)
   - `df2`: River basin geometries (shapefile, 310 basins)
   - `df3`: Water conflict events (primary dataset - participants, issues, outcomes)
   - `df4`: International treaties
2. **EDA & cleaning** (cells ~17-80): Missing data analysis (`missingno`), column pruning, word clouds, geospatial plotting (`geopandas`)
3. **Feature engineering** (cells ~87-230): Merges datasets on basin codes (`BCCODE`), constructs temporal features, aggregates treaty counts
4. **ML classification** (cells ~235-265): Predicts conflict outcome using scikit-learn. Compares ~8 classifiers (RandomForest, KNN, SVM, GaussianProcess, DecisionTree, AdaBoost, MLP, SGD) via accuracy and confusion matrices. Three rounds of model comparison with different feature sets.

## Key Dependencies

Python packages: `pandas`, `numpy`, `geopandas`, `scikit-learn`, `matplotlib`, `missingno`, `wordcloud`, `openpyxl`

System libraries (Colab-specific): `libproj-dev`, `proj-data`, `proj-bin`, `libgeos-dev`

## Data Notes

- Input files are loaded from Google Drive paths (`/content/drive/MyDrive/Colab Notebooks/...`) and are not included in this repo
- The primary join key across datasets is `BCCODE` (basin country code)
- Target variable for ML models is the conflict outcome (ordinal scale from war to cooperation)

## Other Files

- `presentation.pptx`: Project presentation slides
- `tableau_final_project_felix.twb`: Tableau workbook for visualization
