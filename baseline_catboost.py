import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from catboost import CatBoostClassifier

# ======================================
# Create models folder
# ======================================
os.makedirs("models", exist_ok=True)

# ======================================
# Load Dataset
# ======================================
df = pd.read_csv("dataset/engine_fault_detection_dataset.csv")

print("="*60)
print("DATASET LOADED SUCCESSFULLY")
print("="*60)

# ======================================
# Features and Target
# ======================================
X = df.drop("Engine_Condition", axis=1)
y = df["Engine_Condition"]

# ======================================
# Train Test Split
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Samples :", X_train.shape)
print("Testing Samples  :", X_test.shape)

# ======================================
# Improved CatBoost Model
# ======================================
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=8,
    loss_function="MultiClass",
    eval_metric="TotalF1",
    random_seed=42,
    verbose=100
)

# ======================================
# Train
# ======================================
model.fit(X_train, y_train)

# ======================================
# Save Model
# ======================================
joblib.dump(model, "models/baseline_catboost.pkl")

print("\nModel Saved Successfully!")

# ======================================
# Prediction
# ======================================
y_pred = model.predict(X_test)

# ======================================
# Metrics
# ======================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n" + "="*60)
print("BASELINE CATBOOST PERFORMANCE")
print("="*60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))