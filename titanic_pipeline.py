
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
import os

os.makedirs("outputs", exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING & EXPLORATION
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 1: DATA LOADING & EXPLORATION")
print("=" * 60)

df = pd.read_csv("data/titanic.csv")
print(f"\nDataset shape : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Survival rate : {df['Survived'].mean()*100:.1f}% survived")

print("\nMissing values per column:")
missing = df.isnull().sum()
print(missing[missing > 0].to_string())

print("\n[!] Data Note: 'Sex' perfectly predicts 'Survived' in this CSV")
print("    (all males died, all females survived — Kaggle test split artifact).")
print("    Sex is DROPPED to prevent data leakage and ensure realistic scores.")


# ══════════════════════════════════════════════════════════════════
# SECTION 2 — PREPROCESSING
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 2: PREPROCESSING")
print("=" * 60)

# Features (Sex intentionally excluded — see note above)
FEATURES = ["Pclass", "Age", "SibSp", "Parch",
            "Fare", "Embarked", "FamilySize", "IsAlone", "Age_x_Pclass"]
TARGET = "Survived"

def preprocess(df):
    """
    Full preprocessing pipeline.
    - Uses assignment (df[col] = ...) not inplace=True
      (pandas 2.x / Python 3.13 Copy-on-Write compatibility fix)
    - All imputation happens BEFORE feature engineering so
      derived columns (Age_x_Pclass) don't inherit NaNs.
    - Sex is dropped to avoid perfect-score data leakage.
    """
    df = df.copy()

    # Drop irrelevant / leaky columns
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "Sex"])

    # STEP 1: Impute missing values first (before feature engineering)
    age_median  = df["Age"].median()
    fare_median = df["Fare"].median()
    emb_mode    = df["Embarked"].mode()[0]

    df["Age"]      = df["Age"].fillna(age_median)
    df["Fare"]     = df["Fare"].fillna(fare_median)
    df["Embarked"] = df["Embarked"].fillna(emb_mode)

    print(f"  -> Age      : {df['Age'].isna().sum()} NaNs remaining  (filled median = {age_median:.1f})")
    print(f"  -> Fare     : {df['Fare'].isna().sum()} NaNs remaining  (filled median = {fare_median:.2f})")
    print(f"  -> Embarked : {df['Embarked'].isna().sum()} NaNs remaining  (filled mode  = '{emb_mode}')")

    # STEP 2: Encode categorical
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    # STEP 3: Feature engineering (after imputation — no NaN inheritance)
    df["FamilySize"]   = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"]      = (df["FamilySize"] == 1).astype(int)
    df["Age_x_Pclass"] = df["Age"] * df["Pclass"]

    # Final NaN check
    nan_counts = df[FEATURES].isna().sum()
    total_nan  = nan_counts.sum()
    if total_nan > 0:
        print(f"\n  WARNING: {total_nan} NaN(s) still present:")
        print(nan_counts[nan_counts > 0])
    else:
        print(f"\n  Zero NaNs across all {len(FEATURES)} features. Safe to proceed.")

    print(f"  Features: {FEATURES}")
    return df

df_clean = preprocess(df)

X = df_clean[FEATURES]
y = df_clean[TARGET]


# ══════════════════════════════════════════════════════════════════
# SECTION 3 — TRAIN / VALIDATION SPLIT
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 3: TRAIN / VALIDATION SPLIT")
print("=" * 60)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n  Train size      : {len(X_train)} rows ({len(X_train)/len(X)*100:.0f}%)")
print(f"  Validation size : {len(X_val)} rows ({len(X_val)/len(X)*100:.0f}%)")
print(f"  Class balance (train) - Survived: {y_train.mean()*100:.1f}%")
print(f"  Class balance (val)   - Survived: {y_val.mean()*100:.1f}%")

# Scale for Logistic Regression (tree models don't need scaling)
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)


# ══════════════════════════════════════════════════════════════════
# SECTION 4 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 4: MODEL TRAINING")
print("=" * 60)

# Model A: Logistic Regression (linear, interpretable baseline)
lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr.fit(X_train_sc, y_train)
print("\n  [OK] Logistic Regression trained")

# Model B: Random Forest (non-linear, ensemble)
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
print("  [OK] Random Forest trained (200 trees, max_depth=8)")


# ══════════════════════════════════════════════════════════════════
# SECTION 5 — EVALUATION
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 5: EVALUATION")
print("=" * 60)

def evaluate(name, model, X_in, y_true):
    y_pred = model.predict(X_in)
    y_prob = model.predict_proba(X_in)[:, 1]
    return {
        "name"  : name,
        "acc"   : accuracy_score(y_true, y_pred),
        "f1"    : f1_score(y_true, y_pred),
        "auc"   : roc_auc_score(y_true, y_prob),
        "cm"    : confusion_matrix(y_true, y_pred),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

res_lr = evaluate("Logistic Regression", lr, X_val_sc, y_val)
res_rf = evaluate("Random Forest",       rf, X_val,    y_val)

print("\n  +-----------------------+----------+----------+----------+")
print("  | Model                 | Accuracy | F1 Score | AUC-ROC  |")
print("  +-----------------------+----------+----------+----------+")
for r in [res_lr, res_rf]:
    print(f"  | {r['name']:<21} |  {r['acc']:.4f}  |  {r['f1']:.4f}  |  {r['auc']:.4f}  |")
print("  +-----------------------+----------+----------+----------+")

print("\n  Logistic Regression - Classification Report:")
print(classification_report(y_val, res_lr["y_pred"], target_names=["Died", "Survived"]))
print("  Random Forest - Classification Report:")
print(classification_report(y_val, res_rf["y_pred"], target_names=["Died", "Survived"]))


# ══════════════════════════════════════════════════════════════════
# SECTION 6 — FEATURE IMPORTANCE (Random Forest Top 5)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 6: FEATURE IMPORTANCE - Top 5 (Random Forest)")
print("=" * 60)

importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
top5 = importances.head(5)
print("\n  Rank  Feature           Importance")
print("  " + "-" * 40)
for i, (feat, imp) in enumerate(top5.items(), 1):
    bar = "=" * int(imp * 80)
    print(f"  #{i:<4} {feat:<17} {imp:.4f}  {bar}")


# ══════════════════════════════════════════════════════════════════
# SECTION 7 — PREDICTION INTERPRETATION (2 example rows)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 7: PREDICTION INTERPRETATION")
print("=" * 60)

emb_map  = {0: "Southampton", 1: "Cherbourg", 2: "Queenstown"}
idx_surv = y_val[y_val == 1].index[0]
idx_died = y_val[y_val == 0].index[0]

for label, idx in [("Example A", idx_surv), ("Example B", idx_died)]:
    row    = X_val.loc[[idx]]
    actual = y_val.loc[idx]
    prob   = rf.predict_proba(row)[0][1]
    pred   = rf.predict(row)[0]

    emb_val = row["Embarked"].values[0]
    emb_str = emb_map.get(int(emb_val), "Unknown") if not np.isnan(emb_val) else "Unknown"

    pclass_val     = int(row["Pclass"].values[0])
    age_val        = row["Age"].values[0]
    fare_val       = row["Fare"].values[0]
    family_val     = int(row["FamilySize"].values[0])
    alone_val      = row["IsAlone"].values[0]

    outcome_str = "SURVIVED" if pred == 1 else "DIED"
    actual_str  = "Survived" if actual == 1 else "Died"
    alone_str   = "alone" if alone_val == 1 else f"with {family_val - 1} family member(s)"
    fare_level  = "higher" if fare_val > 14 else "lower"
    class_label = {1: "1st", 2: "2nd", 3: "3rd"}.get(pclass_val, "unknown")

    print(f"\n  -- {label} -- Predicted: {outcome_str} --")
    print(f"     Pclass     : {pclass_val}      Fare       : {fare_val:.2f}")
    print(f"     Age        : {age_val:.0f}       FamilySize : {family_val}")
    print(f"     IsAlone    : {'Yes' if alone_val == 1 else 'No'}     Embarked   : {emb_str}")
    print(f"     Actual outcome  : {actual_str}")
    print(f"     Survival prob   : {prob*100:.1f}%")

    print(f"\n     Plain-language interpretation:")
    if pred == 1:
        print(f"     This passenger has a {prob*100:.0f}% survival probability.")
        print(f"     They are a {class_label}-class passenger travelling {alone_str},")
        print(f"     paying a {fare_level} fare ({fare_val:.2f}). These factors")
        print(f"     push the Random Forest prediction toward survival.")
    else:
        print(f"     This passenger has only a {prob*100:.0f}% survival probability.")
        print(f"     They are a {class_label}-class passenger travelling {alone_str},")
        print(f"     paying a {fare_level} fare ({fare_val:.2f}). These factors")
        print(f"     push the Random Forest prediction toward not surviving.")


# ══════════════════════════════════════════════════════════════════
# PLOTS — 5-panel evaluation dashboard
# ══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Titanic ML Pipeline - Evaluation Dashboard",
             fontsize=16, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

COLORS = {"lr": "#4a90d9", "rf": "#27ae60"}

# Plot 1: Confusion Matrix - Logistic Regression
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(res_lr["cm"], annot=True, fmt="d", cmap="Blues", ax=ax1,
            xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"],
            linewidths=0.5, cbar=False)
ax1.set_title("Confusion Matrix\nLogistic Regression", fontweight="bold")
ax1.set_xlabel("Predicted"); ax1.set_ylabel("Actual")

# Plot 2: Confusion Matrix - Random Forest
ax2 = fig.add_subplot(gs[0, 1])
sns.heatmap(res_rf["cm"], annot=True, fmt="d", cmap="Greens", ax=ax2,
            xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"],
            linewidths=0.5, cbar=False)
ax2.set_title("Confusion Matrix\nRandom Forest", fontweight="bold")
ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")

# Plot 3: ROC Curves
ax3 = fig.add_subplot(gs[0, 2])
for res, col, lbl in [(res_lr, COLORS["lr"], "Logistic Reg"),
                       (res_rf, COLORS["rf"], "Random Forest")]:
    fpr, tpr, _ = roc_curve(y_val, res["y_prob"])
    ax3.plot(fpr, tpr, color=col, lw=2, label=f"{lbl} (AUC={res['auc']:.3f})")
ax3.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC=0.5)")
fpr_rf, tpr_rf, _ = roc_curve(y_val, res_rf["y_prob"])
ax3.fill_between(fpr_rf, tpr_rf, alpha=0.07, color=COLORS["rf"])
ax3.set_title("ROC Curves", fontweight="bold")
ax3.set_xlabel("False Positive Rate"); ax3.set_ylabel("True Positive Rate")
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

# Plot 4: Feature Importance
ax4 = fig.add_subplot(gs[1, 0:2])
bar_colors = [COLORS["rf"] if i < 5 else "#bdc3c7" for i in range(len(importances))]
bars = ax4.barh(importances.index[::-1], importances.values[::-1],
                color=bar_colors[::-1], edgecolor="white", height=0.6)
ax4.set_title("Feature Importance - Random Forest (Top 5 highlighted)", fontweight="bold")
ax4.set_xlabel("Importance Score")
for bar, val in zip(bars, importances.values[::-1]):
    ax4.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=8)
ax4.axvline(x=importances.values[4], color="orange", linestyle="--",
            alpha=0.7, linewidth=1.5, label="Top-5 cutoff")
ax4.legend(fontsize=8); ax4.grid(True, axis="x", alpha=0.3)

# Plot 5: Model Comparison Bar Chart
ax5 = fig.add_subplot(gs[1, 2])
metrics = ["Accuracy", "F1 Score", "AUC-ROC"]
lr_vals = [res_lr["acc"], res_lr["f1"], res_lr["auc"]]
rf_vals = [res_rf["acc"], res_rf["f1"], res_rf["auc"]]
x = np.arange(len(metrics))
w = 0.32
b1 = ax5.bar(x - w / 2, lr_vals, w, label="Logistic Reg",  color=COLORS["lr"], edgecolor="white")
b2 = ax5.bar(x + w / 2, rf_vals, w, label="Random Forest", color=COLORS["rf"], edgecolor="white")
for bars_group in [b1, b2]:
    for bar in bars_group:
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)
ax5.set_xticks(x); ax5.set_xticklabels(metrics, fontsize=9)
ax5.set_ylim(0, 1.08); ax5.set_ylabel("Score")
ax5.set_title("Model Comparison", fontweight="bold")
ax5.legend(fontsize=8); ax5.grid(True, axis="y", alpha=0.3)

plt.savefig("outputs/titanic_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Plot saved -> outputs/titanic_evaluation.png")


# ══════════════════════════════════════════════════════════════════
# SECTION 8 — CONCLUSION
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 8: CONCLUSION")
print("=" * 60)

winner = "Random Forest" if res_rf["auc"] >= res_lr["auc"] else "Logistic Regression"
print(f"""
  DEPLOYMENT RECOMMENDATION: {winner}
  {'=' * 50}

  Logistic Regression is fast, interpretable, and a solid
  baseline. It handles linear survival signals (class, fare)
  well with minimal tuning.

  Random Forest captures non-linear interactions
  (e.g., age x class, fare thresholds) and typically edges
  ahead on AUC and F1 on held-out data.

  Top 3 survival drivers (Random Forest):
      1. {importances.index[0]:<16} ({importances.values[0]:.3f}) strongest signal
      2. {importances.index[1]:<16} ({importances.values[1]:.3f})
      3. {importances.index[2]:<16} ({importances.values[2]:.3f})

  Deployment decision:
    - Random Forest     -> best raw performance (AUC / F1)
    - Logistic Regression -> prefer when explainability is
      critical (regulatory/audit: coefficients are directly
      readable and defensible)

  Note: Sex was excluded from features. In this Kaggle CSV,
  Sex perfectly encodes Survived (data artifact of the test
  split). Dropping it yields honest, realistic model scores.
""")

# Save labeled predictions CSV
out = X_val.copy()
out["Actual"]          = y_val.values
out["LR_Predicted"]    = res_lr["y_pred"]
out["LR_Prob_Survive"] = res_lr["y_prob"].round(3)
out["RF_Predicted"]    = res_rf["y_pred"]
out["RF_Prob_Survive"] = res_rf["y_prob"].round(3)
out["Models_Agree"]    = (res_lr["y_pred"] == res_rf["y_pred"])
out.to_csv("outputs/titanic_predictions.csv", index=False)
print("  Predictions saved -> outputs/titanic_predictions.csv")
print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print("=" * 60)