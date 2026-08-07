import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from catboost import CatBoostClassifier


class Fitness:

    def __init__(self):

        # Load Dataset

        self.df = pd.read_csv(
            "dataset/engine_fault_detection_dataset.csv"
        )

        self.X = self.df.drop(
            "Engine_Condition",
            axis=1
        )

        self.y = self.df["Engine_Condition"]

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test

        ) = train_test_split(

            self.X,

            self.y,

            test_size=0.20,

            random_state=42,

            stratify=self.y

        )

    def evaluate(self, hawk):

        model = CatBoostClassifier(

            iterations=hawk["iterations"],

            learning_rate=hawk["learning_rate"],

            depth=hawk["depth"],

            l2_leaf_reg=hawk["l2_leaf_reg"],

            random_strength=hawk["random_strength"],

            bagging_temperature=hawk["bagging_temperature"],

            random_seed=42,

            verbose=False

        )

        model.fit(

            self.X_train,

            self.y_train

        )

        predictions = model.predict(

            self.X_test

        )

        accuracy = accuracy_score(

            self.y_test,

            predictions

        )

        return accuracy