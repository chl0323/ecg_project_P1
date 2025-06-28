# ECG Colab Project

## Overview
This project provides a modular pipeline for ECG (Electrocardiogram) data analysis, including data preprocessing, feature engineering, model training, and continual learning algorithms. The codebase is organized for research, experimentation, and reproducibility.

## Directory Structure
- data_processing_code/: Data preprocessing and feature engineering scripts
- model_code/: Model architectures, training, feature analysis, and explainability
- CL_code/: Continual learning algorithms and evaluation
- requirements.txt: Python dependencies

## Main Components

**Data Processing & Feature Engineering (data_processing_code/):**
- Scripts for data cleaning, feature extraction, and handling missing values.
- Used to preprocess raw ECG data and generate features for modeling.

**Model Training & Analysis (model_code/):**
- Contains model definitions (e.g., transformer, ranpac), training scripts, feature analysis, drift detection, and explainability (e.g., SHAP, attention).
- Used to train models, analyze features, and interpret model predictions.

**Continual Learning (CL_code/):**
- Implements various continual learning strategies (EWC, LwF, iCaRL, GEM, Replay, etc.).
- Includes evaluation and statistical analysis scripts for continual learning experiments.

## Usage
- Use the scripts in data_processing_code/ to preprocess and generate features from raw ECG data.
- Use the scripts in model_code/ to train models, perform feature analysis, and interpret results.
- Use the scripts in CL_code/ to run continual learning experiments and evaluations.

## Notes
- Some scripts may require specific data files to be present in certain folders (e.g., processed data or raw data).
- For more details, refer to the comments and docstrings in each script.

## Contact
For questions or collaboration, please contact the project maintainer.
