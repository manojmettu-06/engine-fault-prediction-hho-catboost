import os
import sys
import time
import subprocess

os.system("cls" if os.name == "nt" else "clear")

print("=" * 70)
print("ENGINE FAULT PREDICTION AND MAINTENANCE DECISION SUPPORT")
print("HARRIS HAWKS OPTIMIZED CATBOOST FRAMEWORK")
print("=" * 70)

modules = [
    ("Data Preprocessing", "preproccesing.py"),
    ("Baseline CatBoost", "baseline_catboost.py"),
    ("Harris Hawks Optimization", "optimize_catboost.py"),
    ("Feature Importance", "explainability.py"),
    ("Maintenance Recommendation", "recommendation.py")
]

for title, script in modules:

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    try:
        subprocess.run(
            [sys.executable, script],
            check=True
        )
        print("✓ Completed Successfully")

    except subprocess.CalledProcessError:
        print(f"❌ Error while executing {script}")
        break

    time.sleep(1)

print("\n" + "=" * 70)
print("PROJECT EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 70)