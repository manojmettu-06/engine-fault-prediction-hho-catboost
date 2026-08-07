import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ============================================
# LOAD DATASET
# ============================================

print("=" * 60)
print("LOADING DATASET...")
print("=" * 60)

df = pd.read_csv("dataset/engine_fault_detection_dataset.csv")

X = df.drop("Engine_Condition", axis=1)

print("Dataset Loaded Successfully!")
print("Shape :", X.shape)

# ============================================
# LOAD MODEL
# ============================================

print("\nLoading Trained CatBoost Model...")

model = joblib.load("models/baseline_catboost.pkl")

print("Model Loaded Successfully!")

# ============================================
# CREATE OUTPUT FOLDER
# ============================================

os.makedirs("outputs", exist_ok=True)

# ============================================
# FEATURE IMPORTANCE
# ============================================

importance = model.get_feature_importance()

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n")
print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)
print(feature_importance)

# ============================================
# SAVE CSV
# ============================================

feature_importance.to_csv(
    "outputs/feature_importance.csv",
    index=False
)

print("\nCSV Saved Successfully!")

# ============================================
# SAVE PLOT
# ============================================

plt.figure(figsize=(10,6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("CatBoost Feature Importance")

plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()   # <-- Important

print("Plot Saved Successfully!")

print("\n")
print("=" * 60)
print("FILES GENERATED")
print("=" * 60)
print("outputs/feature_importance.csv")
print("outputs/feature_importance.png")

print("\nExplainability Module Completed Successfully!")