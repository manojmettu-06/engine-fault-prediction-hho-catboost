import joblib
import pandas as pd

# ============================================
# LOAD DATASET
# ============================================

df = pd.read_csv("dataset/engine_fault_detection_dataset.csv")

X = df.drop("Engine_Condition", axis=1)

# ============================================
# LOAD TRAINED MODEL
# ============================================

model = joblib.load("models/baseline_catboost.pkl")

# ============================================
# SELECT SAMPLE
# ============================================

sample = X.iloc[[0]]

prediction = model.predict(sample)[0]

print("=" * 60)
print("ENGINE HEALTH REPORT")
print("=" * 60)

condition = int(prediction.item())

if condition == 0:
    print("Predicted Engine Condition : NORMAL")
elif condition == 1:
    print("Predicted Engine Condition : WARNING")
else:
    print("Predicted Engine Condition : FAULT")

print()

# ============================================
# FEATURE IMPORTANCE
# ============================================

importance = model.get_feature_importance()

feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("=" * 60)
print("TOP IMPORTANT FEATURES")
print("=" * 60)

print(importance_df.head(5))

print()

# ============================================
# MAINTENANCE RECOMMENDATION
# ============================================

print("=" * 60)
print("MAINTENANCE RECOMMENDATION")
print("=" * 60)

if condition == 0:

    print("✔ Engine operating normally.")
    print("✔ Continue routine maintenance.")
    print("✔ Monitor sensor readings periodically.")
    print("✔ Perform scheduled inspections.")

elif condition == 1:

    print("⚠ Engine showing early warning signs.")
    print("✔ Inspect bearings.")
    print("✔ Check lubrication.")
    print("✔ Inspect vibration levels.")
    print("✔ Monitor exhaust pressure.")
    print("✔ Schedule preventive maintenance.")

else:

    print("❌ Engine fault detected.")
    print("✔ Stop engine if necessary.")
    print("✔ Inspect shaft alignment.")
    print("✔ Replace worn bearings.")
    print("✔ Check intake and exhaust pressure.")
    print("✔ Inspect vibration sensors.")
    print("✔ Perform complete maintenance.")

print()
print("=" * 60)
print("REPORT GENERATED SUCCESSFULLY")
print("=" * 60)