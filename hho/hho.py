from hho.population import Population
from hho.fitness import Fitness
from hho.update import Update


class HarrisHawksOptimization:

    def __init__(

        self,

        population_size,

        max_iterations,

        bounds

    ):

        self.population_size = population_size

        self.max_iterations = max_iterations

        self.bounds = bounds

        self.population = Population(

            population_size,

            bounds

        ).initialize()

        self.fitness = Fitness()

        self.update = Update(bounds)

        self.rabbit = None

        self.rabbit_score = -1

    # ======================================
    # Evaluate Population
    # ======================================

    def evaluate(self):

        for hawk in self.population:

            score = self.fitness.evaluate(hawk)

            if score > self.rabbit_score:

                self.rabbit_score = score

                self.rabbit = hawk.copy()

    # ======================================
    # Optimization
    # ======================================

    def optimize(self):

        print("=" * 70)

        print("HARRIS HAWKS OPTIMIZATION STARTED")

        print("=" * 70)

        for iteration in range(self.max_iterations):

            self.evaluate()

            print(

                f"Iteration {iteration+1:02d}"

                f"  Best Accuracy : {self.rabbit_score:.4f}"

            )

            new_population = []

            for hawk in self.population:

                updated = self.update.update_hawk(

                    hawk,

                    self.rabbit,

                    iteration,

                    self.max_iterations

                )

                new_population.append(updated)

            self.population = new_population

        print("\n")

        print("=" * 70)

        print("FINAL RABBIT")

        print("=" * 70)

        print(self.rabbit)

        print()

        print("Best Accuracy :", round(self.rabbit_score,4))

        return self.rabbit, self.rabbit_score