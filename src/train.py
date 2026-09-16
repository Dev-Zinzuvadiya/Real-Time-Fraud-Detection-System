import os
import joblib
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# FILE PATHS
DATA_FILE = "data/processed/fraud_processed.csv"
MODEL_FILE = "models/fraud_model.pkl"


# LOAD DATA
print(">> LOADING PROCESSED DATASET...")
df = pd.read_csv(DATA_FILE)

print(">> PROCESSED DATASET LOADED !!")
print("\n:> DATASET SHAPE:", df.shape)


# SEPARATE FEATURES AND TARGET
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

print("\n:> FEATURES:", X.shape)
print(":> TARGET:", y.shape)


# REMOVE ID COLUMNS
id_columns = []

for column in X.columns:
    column_lower = column.lower()
    if (
        column == "id"
        or column_lower.endswith("_id")
        or column_lower == "transaction_id"
    ):
        id_columns.append(column)

if len(id_columns) > 0:
    print(f"\n:> REMOVING ID COLUMNs: {id_columns}")
    X = X.drop(columns=id_columns)


# IDENTIFY NUMERICAL FEATURES & CATEGORICAL FEATURES
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print("\n\n>> NUMERICAL FEATUREs ↘")
print("-" * 30)
for column in numeric_features:
    print("-", column)


print("\n\n>> CATEGORICAL FEATUREs ↘")
print("-" * 30)
for column in categorical_features:
    print("-", column)


# TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\n:> TRAINING ROWs:", len(X_train))
print(":> TESTING ROWs:", len(X_test))


# PREPROCESSING
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        )
    ],
    remainder="passthrough",
)


# MODEL
n_trees = 100
model = RandomForestClassifier(
    n_estimators=1, warm_start=True, random_state=42, n_jobs=1
)

# COMPLETE ML PIPELINE
pipeline = Pipeline(steps=[("preprocessing", preprocessor), ("model", model)])

# TRAIN MODEL
print("\n>> TRAINING RANDOM-FOREST...")
print("-" * 35)

with tqdm(total=n_trees, desc="-> TRAINING TREEs", unit="tree") as pbar:
    for i in range(1, n_trees + 1):
        pipeline.named_steps["model"].set_params(n_estimators=i)
        pipeline.fit(X_train, y_train)
        pbar.update(1)

print(">> MODEL TRAINING COMPLETED!!")


# PREDICTIONS
print("\n>> MAKING PREDICTIONs...")
y_pred = pipeline.predict(X_test)


# EVALUATION
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)


# PRINT METRICS
print("\n==============================")
print("MODEL EVALUATION")
print("==============================")
print(f":> Accuracy : {accuracy:.4f}")
print(f":> Precision: {precision:.4f}")
print(f":> Recall   : {recall:.4f}")
print(f":> F1 Score : {f1:.4f}")


# CLASSIFICATION REPORT
print("\n>> CLASSIFICATION REPORT ↘")
print("-" * 30)
print(classification_report(y_test, y_pred, zero_division=0))


# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
print("\n>> CONFUSION MATRIX ↘")
print(cm)


# CONFUSION MATRIX DISPLAY
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("Fraud Detection Confusion Matrix")
plt.tight_layout()
plt.show()


# CREATE MODEL DIRECTORY
os.makedirs("models", exist_ok=True)


# SAVE COMPLETE PIPELINE
joblib.dump(pipeline, MODEL_FILE)
print(f"\n:> COMPLETE MODEL PIPELINE SAVED TO: {MODEL_FILE}")


# FINAL MESSAGE
print("\n>> TRAINING PROCESS COMPLETED SUCCESSFULLY !!...")
