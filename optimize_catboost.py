from hho.hho import HarrisHawksOptimization

# ===============================================
# SEARCH SPACE
# ===============================================

BOUNDS = {

    "iterations": (50,150),

    "learning_rate": (0.05,0.20),

    "depth": (4,8),

    "l2_leaf_reg": (1,6),

    "random_strength": (1,3),

    "bagging_temperature": (0,2)

}

# ===============================================
# RUN HHO
# ===============================================

optimizer = HarrisHawksOptimization(

    population_size=5,

    max_iterations=10,

    bounds=BOUNDS

)

best_hawk, best_accuracy = optimizer.optimize()

print("\n")

print("="*70)

print("OPTIMIZATION FINISHED")

print("="*70)

print(best_hawk)

print()

print("Best Accuracy :",round(best_accuracy,4))