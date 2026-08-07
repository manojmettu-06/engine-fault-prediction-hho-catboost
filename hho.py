import random
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

# Load Dataset
df = pd.read_csv("dataset/engine_fault_detection_dataset.csv")

X = df.drop("Engine_Condition", axis=1)
y = df["Engine_Condition"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ===============================
# FAST TEST SETTINGS
# ===============================

NUM_HAWKS = 5

hawks = []

for i in range(NUM_HAWKS):

    hawk = {

        "iterations": random.randint(50,150),

        "learning_rate": round(random.uniform(0.05,0.20),3),

        "depth": random.randint(4,8),

        "l2_leaf_reg": random.randint(1,6),

        "random_strength": round(random.uniform(1,3),2),

        "bagging_temperature": round(random.uniform(0,2),2)

    }

    hawks.append(hawk)

print("="*60)
print("HARRIS HAWKS FITNESS EVALUATION")
print("="*60)

best_accuracy = 0
best_hawk = None

for i, hawk in enumerate(hawks):

    print(f"\nEvaluating Hawk {i+1}...")

    model = CatBoostClassifier(

        iterations=hawk["iterations"],

        learning_rate=hawk["learning_rate"],

        depth=hawk["depth"],

        l2_leaf_reg=hawk["l2_leaf_reg"],

        random_strength=hawk["random_strength"],

        bagging_temperature=hawk["bagging_temperature"],

        random_seed=42,

        verbose=0

    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(hawk)
    print("Accuracy :", round(accuracy,4))

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_hawk = hawk

print("\n" + "="*60)
print("BEST HAWK")
print("="*60)

print(best_hawk)
print("Best Accuracy :", round(best_accuracy,4))