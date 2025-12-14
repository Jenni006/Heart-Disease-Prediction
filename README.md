# Heart Disease Prediction

A hands-on exploratory notebook and starter pipeline for predicting cardiovascular disease from a tabular health dataset. This repository contains the Jupyter/Colab notebook `heartdiseaseprediction.ipynb` which walks through loading the dataset, basic data cleaning, exploratory data analysis (EDA), feature preparation and a train/test split — ready for building and evaluating machine learning models.

The notebook was developed and executed in Google Colab and is suitable to run locally as well.

---

Table of contents
- Project summary
- Dataset
- Notebook overview
- Key data preprocessing steps (implemented)
- Suggested next steps / model ideas
- How to run
  - Run in Google Colab
  - Run locally
- Required packages / environment
- Notes, caveats and data quality issues
- Contributing
- License
- Contact

---

Project summary
---------------
The goal of this project is to build a predictive model that determines the presence of cardiovascular disease for individuals given health-related features (e.g., age, gender, height, weight, blood pressure, cholesterol, glucose, lifestyle indicators). The included notebook demonstrates data ingestion, basic cleaning, visualization of feature correlations, and splitting data into training and test sets.

Dataset
-------
- The notebook expects a CSV file with a semicolon delimiter (`;`). The dataset contains the following columns (as used in the notebook):
  - id
  - age (stored as integer; in this dataset values appear to be in days)
  - gender (1, 2)
  - height (cm)
  - weight (kg)
  - ap_hi (systolic blood pressure)
  - ap_lo (diastolic blood pressure)
  - cholesterol (categorical: 1, 2, 3)
  - gluc (categorical: 1, 2, 3)
  - smoke (0/1)
  - alco (0/1)
  - active (0/1)
  - cardio (0/1) — target variable

Notebook overview
-----------------
File: `heartdiseaseprediction.ipynb` (the primary notebook in this repo)

High-level steps performed in the notebook:
1. Imports: pandas, numpy, seaborn, matplotlib, scikit-learn utilities.
2. Data upload (Colab): uses `files.upload()` and reads the CSV using `pd.read_csv(..., delimiter=';')`.
3. Quick inspection: `df.head()` and `df.describe()` to get column stats.
4. Missing value handling: fills NA with the column median (`df.fillna(df.median(), inplace=True)`).
5. Correlation heatmap: visualize feature correlations using seaborn.
6. Feature/target split:
   - X = numeric features (all numeric columns except the last one)
   - y = last numeric column (assumed to be `cardio`)
7. Train/test split: `train_test_split` with `test_size=0.2, random_state=42`.
8. Shapes observed in the run (example outputs recorded in notebook): total numeric matrix shape (70000, 12), training shape (56000, 12), test shape (14000, 12).

Key data preprocessing steps (implemented)
------------------------------------------
- Read CSV with semicolon delimiter.
- Filled missing values with median per column.
- Selected only numeric columns for modeling convenience (categorical encoded as numbers already in dataset).
- Target column assumed to be the last numeric column (named `cardio` in the dataset).
- Train/test split (80/20) with a fixed random seed for reproducibility.

Suggested next steps / model ideas
---------------------------------
The notebook currently prepares data and performs EDA. Recommended next steps:
- Data cleaning improvements
  - Convert `age` from days to years (e.g., age_years = age / 365).
  - Remove or impute unrealistic outliers (e.g., negative blood pressure values or extremely large values).
  - Cap or remove physically impossible values (ap_hi < ap_lo).
  - Drop the `id` column (not predictive).
- Feature engineering
  - Compute BMI: weight (kg) / (height (m))^2 and consider using BMI instead of raw height/weight.
  - One-hot encode categorical features if necessary (cholesterol, gluc) or treat as ordinal if appropriate.
  - Interaction features (e.g., age*bmi) and normalization/scaling for some models.
- Modeling
  - Baseline models: Logistic Regression, Decision Tree, Random Forest.
  - Advanced models: XGBoost / LightGBM, CatBoost.
  - Consider cross-validation and hyperparameter tuning (GridSearchCV / RandomizedSearchCV).
- Evaluation
  - Use metrics appropriate for the classification task: accuracy, precision, recall, F1-score, ROC AUC, confusion matrix.
  - Use stratified splits or CV if the target is imbalanced.
- Explainability and monitoring
  - Use SHAP or permutation importance to explain predictions.
  - Save the best model (joblib / pickle) and add inference example.

How to run
----------
Run in Google Colab (recommended for reproducibility):
1. Open the notebook `heartdiseaseprediction.ipynb` in Colab (File -> Open notebook -> GitHub and paste the repository URL).
2. Upload the CSV when prompted by `files.upload()` or mount Google Drive and read the dataset from Drive.
3. Execute cells top-to-bottom.

Run locally (Jupyter notebook / Jupyter Lab):
1. Clone the repository:
   git clone https://github.com/Jenni006/Heart-Disease-Prediction.git
2. Create and activate a virtual environment (recommended):
   python -m venv venv
   source venv/bin/activate  # Linux / macOS
   venv\Scripts\activate     # Windows
3. Install required packages:
   pip install -r requirements.txt
   (If `requirements.txt` is not present, install these:)
   pip install pandas numpy seaborn matplotlib scikit-learn jupyter
4. Start Jupyter:
   jupyter notebook
5. Open `heartdiseaseprediction.ipynb`, upload the dataset (or place it in the repo folder) and run cells.

Required packages / environment
------------------------------
Minimum recommended packages:
- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
Optional:
- google-colab (for Colab-specific helper calls)
- jupyter / notebook / jupyterlab

You can create a requirements.txt including:
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter

Notes, caveats and data quality issues
-------------------------------------
- The dataset in the notebook contains anomalous values (e.g., negative blood pressure values and extremely large maxima for ap_hi/ap_lo) — these likely represent data entry errors. It is crucial to detect and handle these before training models.
- `age` appears to be in days rather than years. Convert to years for interpretability.
- The notebook currently uses numeric selection for features and assumes the last numeric column is the target. If your dataset layout differs, update the selection logic accordingly.
- Always validate preprocessing on a holdout set and avoid leaking information from the test set into preprocessing steps (e.g., scaling should be fit on training data only).

Contributing
------------
Contributions are welcome! Suggested ways to contribute:
- Add model training and evaluation cells to the notebook (e.g., logistic regression baseline, comparison of multiple models).
- Add a `requirements.txt` and a script for training (e.g., `train.py`) to reproduce experiments headlessly.
- Improve data cleaning and document assumptions.
- Add unit tests or data validation scripts.

License
-------
This repository does not include a license file by default. If you want to release it publicly, consider adding an open source license (MIT, Apache-2.0, etc.) and include a `LICENSE` file.

Contact
-------
For questions or suggestions, open an issue in the repository or reach out to the repository owner.

Acknowledgements
----------------
This notebook was implemented and executed in Google Colab. The dataset structure used here is commonly seen in public heart-disease / cardiovascular prediction challenges — ensure you cite the original dataset source if one is available when publishing results based on it.
