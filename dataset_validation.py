import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("dataset/engine_fault_detection_dataset.csv")

print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(df.shape)

print("\nCLASS DISTRIBUTION")
print(df["Engine_Condition"].value_counts())

print("\nCORRELATION WITH TARGET")
corr = df.corr(numeric_only=True)

print(corr["Engine_Condition"].sort_values(ascending=False))

plt.figure(figsize=(12,6))
corr["Engine_Condition"].sort_values().plot(kind="barh")
plt.title("Feature Correlation with Engine Condition")
plt.xlabel("Correlation")
plt.tight_layout()
plt.show()