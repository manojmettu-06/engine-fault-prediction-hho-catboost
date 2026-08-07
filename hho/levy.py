import math
import random
import numpy as np

# ==========================================================
# LEVY FLIGHT FUNCTION
# ==========================================================

def levy_flight(beta=1.5):

    sigma = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (
            math.gamma((1 + beta) / 2)
            * beta
            * (2 ** ((beta - 1) / 2))
        )
    ) ** (1 / beta)

    u = np.random.normal(0, sigma)
    v = np.random.normal(0, 1)

    step = u / (abs(v) ** (1 / beta))

    return step