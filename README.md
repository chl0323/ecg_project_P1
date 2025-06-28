Main Components
1. Data Processing & Feature Engineering (data_processing_code/)
Scripts for data cleaning, feature extraction, and handling missing values.
Typical usage: preprocess raw ECG data and generate features for modeling.
2. Model Training & Analysis (model_code/)
Contains model definitions (e.g., transformer, ranpac), training scripts, feature analysis, drift detection, and explainability (e.g., SHAP, attention).
Use these scripts to train models, analyze features, and interpret model predictions.
3. Continual Learning (CL_code/)
Implements various continual learning strategies (EWC, LwF, iCaRL, GEM, Replay, etc.).
Includes evaluation and statistical analysis scripts for continual learning experiments.
Environment Setup
Clone the repository
Apply to data_process...
Run
>
Install dependencies
It is recommended to use a virtual environment:
Apply to data_process...
Run
txt
Usage
Data Processing:
Run scripts in data_processing_code/ to preprocess and generate features from raw ECG data.
Model Training & Analysis:
Use scripts in model_code/ to train models, perform feature analysis, and interpret results.
Continual Learning:
Run continual learning experiments and evaluations using scripts in CL_code/.
Notes
Some scripts may require specific data files to be present in certain folders (e.g., processed data or raw data).
For more details, refer to the comments and docstrings in each script.
Contact
For questions or collaboration, please contact the project maintainer.
