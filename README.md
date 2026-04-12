# Titanic Survival — Tabular ML Pipeline

> **Task 4 (Day 2, ML on Tabular Data)** · End-to-end classification pipeline with preprocessing, modeling, evaluation, and explainability  
> Stack: Python · Pandas · Scikit-learn · Matplotlib · Seaborn

---

## Project Structure

```
titanic_ml/
├── data/
│   └── titanic.csv                  # Input dataset (Kaggle Titanic)
├── outputs/
│   ├── titanic_evaluation.png       # 5-panel evaluation dashboard
│   └── titanic_predictions.csv      # Validation rows with predictions
├── titanic_pipeline.py              # Main pipeline script
├── requirements.txt                 # Python dependencies
└── README.md                        # You are here
```

---

## Dataset

| Field | Detail |
|---|---|
| Source | Kaggle Titanic Competition |
| File | `titanic.csv` |
| Rows | 418 passengers |
| Target | `Survived` (0 = Died, 1 = Survived) |
| Class balance | ~36.4% survived |

### Important Data Note

This CSV is Kaggle's **test split**, not the training set. In it, every male passenger died and every female survived — a known artifact of how this particular file was constructed. Using `Sex` as a feature causes **perfect 1.0 scores** (data leakage), which is not a real model result.

**Fix applied:** `Sex` is dropped from features. The pipeline uses `Pclass`, `Age`, `Fare`, `FamilySize`, `IsAlone`, `Embarked`, and `Age_x_Pclass` instead — giving honest, meaningful model scores.

---

## Setup

### 1. Create the project folder

```bash
mkdir titanic_ml && cd titanic_ml
mkdir data outputs
```

### 2. Place your data

Copy `titanic.csv` into the `data/` folder.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

### 4. Run the pipeline

```bash
python titanic_pipeline.py
```

---

## Pipeline Sections

### Section 1 — Data Loading & Exploration

Loads the CSV, prints shape, survival rate, and missing value counts per column. Also flags the `Sex` leakage issue upfront so it is transparent and documented.

### Section 2 — Preprocessing

All steps follow a strict order to avoid NaN inheritance in derived features:

| Step | Detail |
|---|---|
| Drop columns | `PassengerId`, `Name`, `Ticket`, `Cabin`, `Sex` |
| Impute Age | Median (robust to skewed distribution) |
| Impute Fare | Median (only 1 missing value) |
| Impute Embarked | Mode (most frequent port) |
| Encode Embarked | S=0, C=1, Q=2 |
| Engineer FamilySize | SibSp + Parch + 1 |
| Engineer IsAlone | 1 if FamilySize == 1 |
| Engineer Age_x_Pclass | Age x Pclass interaction term |

**Why assignment over inplace:** `df["col"] = df["col"].fillna(...)` is used throughout instead of `.fillna(inplace=True)`. Pandas 2.x (Python 3.13) Copy-on-Write semantics make `inplace=True` silently fail on column slices — the NaN fix never applies and values stay missing, causing the `Input X contains NaN` error at model fit time.

### Section 3 — Train / Validation Split

80/20 stratified split (`stratify=y`) ensures both splits have the same ~36% survival rate. Features are scaled with `StandardScaler` for Logistic Regression only — tree-based models are scale-invariant and do not need it.

### Section 4 — Model Training

Two models are trained and compared:

**Logistic Regression** (linear baseline)
- `C=1.0`, `max_iter=1000`
- Trained on scaled features
- Fast, interpretable, coefficients directly readable

**Random Forest** (non-linear ensemble)
- 200 trees, `max_depth=8`, `min_samples_leaf=5`
- Trained on raw (unscaled) features
- Captures non-linear interactions and feature combinations

### Section 5 — Evaluation

Three metrics are reported:

| Metric | Why it matters here |
|---|---|
| Accuracy | Overall correctness — easy to communicate |
| F1 Score | Balances precision and recall — important with class imbalance (~36% survived) |
| AUC-ROC | Threshold-independent ranking quality — best single metric for binary classifiers |

Accuracy alone is misleading here: a model that always predicts "died" would score ~63% without learning anything. F1 and AUC expose this failure mode.

### Section 6 — Feature Importance (Top 5)

Uses Random Forest's built-in `feature_importances_` (Gini impurity reduction). The top 5 features are printed with a text bar chart and highlighted in the dashboard plot. This reveals which signals the model relies on most — typically Fare, Pclass, Age, and FamilySize after Sex is excluded.

### Section 7 — Prediction Interpretation

Two real validation rows are decoded in plain language — one predicted survivor and one predicted to die. For each row the script prints:

- All feature values (class, age, fare, family size, embarkation port)
- Actual vs. predicted outcome
- Survival probability from the Random Forest
- A plain-English explanation of which factors drove the prediction

This makes the model's reasoning accessible to non-technical stakeholders without needing SHAP or LIME.

### Section 8 — Conclusion

Prints a deployment recommendation with reasoning, top 3 feature importances, and guidance on when to prefer each model.

---

## Outputs

### `outputs/titanic_evaluation.png`

A 5-panel dashboard:

| Panel | Content |
|---|---|
| Top-left | Confusion matrix — Logistic Regression |
| Top-center | Confusion matrix — Random Forest |
| Top-right | ROC curves for both models with AUC scores |
| Bottom-left/center | Feature importance bar chart (top 5 highlighted in green) |
| Bottom-right | Side-by-side Accuracy / F1 / AUC comparison |

### `outputs/titanic_predictions.csv`

Every validation row labeled with both models' predictions and probabilities:

| Column | Description |
|---|---|
| Feature columns | All 9 preprocessed input features |
| `Actual` | True survival label (0/1) |
| `LR_Predicted` | Logistic Regression prediction (0/1) |
| `LR_Prob_Survive` | LR survival probability (0.0–1.0) |
| `RF_Predicted` | Random Forest prediction (0/1) |
| `RF_Prob_Survive` | RF survival probability (0.0–1.0) |
| `Models_Agree` | True if both models predict the same outcome |

Rows where `Models_Agree = False` are the most uncertain passengers — worth inspecting manually as borderline cases.

---

## Model Comparison

| | Logistic Regression | Random Forest |
|---|---|---|
| Type | Linear, parametric | Ensemble, non-linear |
| Training data | Scaled features | Raw features |
| Interpretability | High (coefficients) | Medium (feature importances) |
| Handles interactions | No | Yes |
| Speed | Very fast | Fast |
| Best for | Explainability-first contexts | Maximum predictive performance |

**Deployment recommendation:** Random Forest for best AUC/F1. Logistic Regression if the model needs to be auditable or explained to regulators — its coefficients map directly to feature contributions with no ambiguity.

---

## Known Limitations

**Small dataset:** 418 rows gives noisy validation metrics. A single bad split can shift AUC by several points. Cross-validation (`StratifiedKFold`) would give more stable estimates on a dataset this size.

**Test-split artifact:** The `Sex` leakage issue is unique to this CSV. On the full Titanic training dataset, Sex is a legitimate and important feature — just not usable here without artificially inflating scores to 1.0.

**No hyperparameter tuning:** Fixed hyperparameters are used for clarity. A `RandomizedSearchCV` pass over `n_estimators`, `max_depth`, and `min_samples_leaf` would likely improve Random Forest AUC further.

---

## Troubleshooting

**`ValueError: Input X contains NaN`**
Your pandas version uses Copy-on-Write (pandas 2.x / Python 3.13). Make sure you are using the latest script which uses `df["col"] = df["col"].fillna(...)` instead of `.fillna(inplace=True)`.

**`ValueError: Invalid format specifier`**
Python f-string width and precision order must be `:<width>.<precision><type>` — for example `{value:<10.0f}` not `{value:.0f:<10}`. The latest script has this corrected throughout.

**Perfect 1.0000 scores across all metrics**
The `Sex` column is leaking the target. Check that `"Sex"` is included in the `drop(columns=[...])` call inside `preprocess()`.

**`FileNotFoundError: data/titanic.csv`**
The CSV must be inside the `data/` subfolder, not in the project root. Create the subfolder and move the file: `mkdir data && mv titanic.csv data/`.

---

## Author

Built as part of **Day 2 Task 4** — demonstrating structured ML thinking on a tabular classification problem: preprocessing, dual-model training, multi-metric evaluation, and lightweight explainability.