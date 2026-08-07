import random
import copy

from hho.levy import levy_flight


class Update:

    def __init__(self, bounds):

        self.bounds = bounds

    # ==========================================
    # Escape Energy
    # ==========================================

    def escape_energy(self, iteration, max_iterations):

        E0 = random.uniform(-1, 1)

        E = 2 * E0 * (1 - iteration / max_iterations)

        return E

    # ==========================================
    # Update One Hawk
    # ==========================================

    def update_hawk(self, hawk, rabbit, iteration, max_iterations):

        new_hawk = copy.deepcopy(hawk)

        E = self.escape_energy(iteration, max_iterations)

        for parameter in hawk:

            lower = self.bounds[parameter][0]
            upper = self.bounds[parameter][1]

            # --------------------------------------
            # Exploration
            # --------------------------------------

            if abs(E) >= 1:

                value = hawk[parameter] + random.random() * (
                    rabbit[parameter] - hawk[parameter]
                )

            # --------------------------------------
            # Exploitation
            # --------------------------------------

            else:

                jump = levy_flight()

                value = rabbit[parameter] - E * abs(
                    jump * rabbit[parameter] - hawk[parameter]
                )

            # --------------------------------------
            # Boundary Control
            # --------------------------------------

            value = max(lower, min(upper, value))

            if parameter in [

                "iterations",

                "depth",

                "l2_leaf_reg"

            ]:

                value = int(round(value))

            else:

                value = round(value, 3)

            new_hawk[parameter] = value

        return new_hawk