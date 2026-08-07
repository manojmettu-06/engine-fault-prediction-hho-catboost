import random

# ==========================================================
# HARRIS HAWKS POPULATION INITIALIZATION
# ==========================================================

class Population:

    def __init__(self, population_size, bounds):

        self.population_size = population_size
        self.bounds = bounds

    def initialize(self):

        hawks = []

        for _ in range(self.population_size):

            hawk = {

                "iterations": random.randint(
                    self.bounds["iterations"][0],
                    self.bounds["iterations"][1]
                ),

                "learning_rate": round(
                    random.uniform(
                        self.bounds["learning_rate"][0],
                        self.bounds["learning_rate"][1]
                    ),
                    3
                ),

                "depth": random.randint(
                    self.bounds["depth"][0],
                    self.bounds["depth"][1]
                ),

                "l2_leaf_reg": random.randint(
                    self.bounds["l2_leaf_reg"][0],
                    self.bounds["l2_leaf_reg"][1]
                ),

                "random_strength": round(
                    random.uniform(
                        self.bounds["random_strength"][0],
                        self.bounds["random_strength"][1]
                    ),
                    2
                ),

                "bagging_temperature": round(
                    random.uniform(
                        self.bounds["bagging_temperature"][0],
                        self.bounds["bagging_temperature"][1]
                    ),
                    2
                )

            }

            hawks.append(hawk)

        return hawks