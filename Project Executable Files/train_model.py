"""
Smart Lender - Loan Prediction Model
Covers Epic 2 (EDA), Epic 3 (Preprocessing), Epic 4 (Model Building)

BEFORE RUNNING:
1. Download the dataset from Kaggle: "Loan Prediction Problem Dataset"
   (or the source your mentor gave you) and save it as:
   smart_lender/loan_data.csv
2. pip install pandas numpy scikit-learn matplotlib seaborn xgboost joblib
3. Run: python train_model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

sns.set(style="whitegrid")
os.makedirs("plots", exist_ok=True)

# ============================================================
# EPIC 2, STORY 1: Import and read the dataset
# ============================================================
df = pd.read_csv("loan_data.csv")
print("Shape:", df.shape)
print(df.head())
print(df.info())

# ============================================================
# EPIC 2, STORY 2: Univariate analysis
# ============================================================
plt.figure(figsize=(6, 4))
sns.countplot(x="Loan_Status", data=df)
plt.title("Loan Status Distribution")
plt.savefig("plots/univariate_loan_status.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.histplot(df["ApplicantIncome"], kde=True)
plt.title("Applicant Income Distribution")
plt.savefig("plots/univariate_income.png")
plt.close()

# ============================================================
# EPIC 2, STORY 3: Bivariate analysis
# ============================================================
plt.figure(figsize=(6, 4))
sns.countplot(x="Education", hue="Loan_Status", data=df)
plt.title("Education vs Loan Status")
plt.savefig("plots/bivariate_education_loan.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.boxplot(x="Loan_Status", y="ApplicantIncome", data=df)
plt.title("Income vs Loan Status")
plt.savefig("plots/bivariate_income_loan.png")
plt.close()

# ============================================================
# EPIC 2, STORY 4: Multivariate analysis
# ============================================================
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("plots/multivariate_correlation.png")
plt.close()

print("EDA plots saved in /plots folder.")

# ============================================================
# EPIC 3, STORY 1: Handle missing values, duplicates
# ============================================================
df.drop_duplicates(inplace=True)

for col in ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History"]:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in ["LoanAmount", "Loan_Amount_Term"]:
    df[col].fillna(df[col].median(), inplace=True)

df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)

# Drop ID column, not useful for prediction
if "Loan_ID" in df.columns:
    df.drop("Loan_ID", axis=1, inplace=True)

# Encode categorical columns
label_encoders = {}
categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

joblib.dump(label_encoders, "label_encoders.pkl")

# ============================================================
# EPIC 3, STORY 4: Split into X / y, train / test
# (Split BEFORE balancing/scaling to avoid leakage)
# ============================================================
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# EPIC 3, STORY 2: Balance the dataset (SMOTE on training set only)
# ============================================================
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print("Before balancing:", y_train.value_counts().to_dict())
print("After balancing:", y_train_bal.value_counts().to_dict())

# ============================================================
# EPIC 3, STORY 3: Scale numerical features
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")

# ============================================================
# EPIC 4: Model Building - train all 4 models
# ============================================================
results = {}

# Story 1: Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train_bal)
results["Decision Tree"] = {
    "train_acc": accuracy_score(y_train_bal, dt.predict(X_train_scaled)),
    "test_acc": accuracy_score(y_test, dt.predict(X_test_scaled)),
    "model": dt,
}

# Story 2: Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train_bal)
results["Random Forest"] = {
    "train_acc": accuracy_score(y_train_bal, rf.predict(X_train_scaled)),
    "test_acc": accuracy_score(y_test, rf.predict(X_test_scaled)),
    "model": rf,
}

# Story 3: KNN
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_scaled, y_train_bal)
results["KNN"] = {
    "train_acc": accuracy_score(y_train_bal, knn.predict(X_train_scaled)),
    "test_acc": accuracy_score(y_test, knn.predict(X_test_scaled)),
    "model": knn,
}

# Story 4: XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
xgb.fit(X_train_scaled, y_train_bal)
results["XGBoost"] = {
    "train_acc": accuracy_score(y_train_bal, xgb.predict(X_train_scaled)),
    "test_acc": accuracy_score(y_test, xgb.predict(X_test_scaled)),
    "model": xgb,
}

print("\n=== Model Comparison ===")
for name, res in results.items():
    print(f"{name}: Train Acc = {res['train_acc']:.3f} | Test Acc = {res['test_acc']:.3f}")

# Pick best model by test accuracy
best_name = max(results, key=lambda k: results[k]["test_acc"])
best_model = results[best_name]["model"]
print(f"\nBest model: {best_name} (Test Acc: {results[best_name]['test_acc']:.3f})")

# Save best model + column order for Flask app
joblib.dump(best_model, "model.pkl")
joblib.dump(list(X.columns), "columns.pkl")
print("\nSaved model.pkl, scaler.pkl, label_encoders.pkl, columns.pkl")
