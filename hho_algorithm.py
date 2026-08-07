import random
import copy

# ============================================
# Escape Energy
# ============================================

def calculate_escape_energy(iteration, max_iterations):

    E0 = random.uniform(-1, 1)

    E = 2 * E0 * (1 - iteration / max_iterations)

    return E


# ============================================
# Update Hawk Position
# ============================================

def update_hawk_position(hawk, rabbit, bounds, E):

    new_hawk = copy.deepcopy(hawk)

    for parameter in hawk:

        lower = bounds[parameter][0]
        upper = bounds[parameter][1]

        # Exploration Phase
        if abs(E) >= 1:

            value = hawk[parameter] + random.uniform(-1,1) * (
                rabbit[parameter] - hawk[parameter]
            )

        # Exploitation Phase
        else:

            value = rabbit[parameter] - E * abs(
                rabbit[parameter] - hawk[parameter]
            )

        # Keep within limits

        value = max(lower, min(upper, value))

        # Integer parameters

        if parameter in ["iterations","depth","l2_leaf_reg"]:

            value = int(round(value))

        else:

            value = round(value,3)

        new_hawk[parameter] = value

    return new_hawk


# ============================================
# Update Entire Population
# ============================================

def update_population(hawks, rabbit, bounds, iteration, max_iterations):

    new_population = []

    E = calculate_escape_energy(iteration, max_iterations)

    for hawk in hawks:

        updated = update_hawk_position(
            hawk,
            rabbit,
            bounds,
            E
        )

        new_population.append(updated)

    return new_population