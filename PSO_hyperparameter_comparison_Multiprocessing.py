import numpy as np
import multiprocessing as mp
from functools import partial
from DemandPrediction_v2 import DemandPrediction
from Main_PSO_v4 import pso

# Define cost function
def cost_function(x):
    training_problem = DemandPrediction("train")
    error = training_problem.evaluate(x)
    return error

def pso_with_hyperparameters(hyperparams):
    c1, c2, w = hyperparams
    cost, pos = pso(cost_function, bounds, n_particles, n_iterations, c1, c2, w, False)
    return (cost, pos, hyperparams)

bounds = np.array(DemandPrediction.bounds())

# Set ranges for hyperparameter tuning
c1_range = [0.3, 0.5, 0.7]
c2_range = [0.3, 0.5, 0.7]
w_range = [0.5, 0.7, 0.9]

n_particles = 100
n_iterations = 1000

# Prepare combinations of hyperparameters
hyperparameter_combinations = [(c1, c2, w) for c1 in c1_range for c2 in c2_range for w in w_range]

def main():
    # Run the multiprocessing pool
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(pso_with_hyperparameters, hyperparameter_combinations)

    # Find the best solution and associated hyperparameters
    best_cost = float('inf')
    best_pos = None
    best_hyperparameters = None

    for cost, pos, hyperparams in results:
        print("c1: {}, c2: {}, w: {}, cost: {}".format(hyperparams[0], hyperparams[1], hyperparams[2], cost))
        if cost < best_cost:
            best_cost = cost
            best_pos = pos
            best_hyperparameters = hyperparams

    print("\nBest hyperparameters found:")
    print("c1: {}, c2: {}, w: {}".format(best_hyperparameters[0], best_hyperparameters[1], best_hyperparameters[2]))
    print("Best training error: {}".format(best_cost))

    # Evaluate the best solution on the test dataset
    test_problem = DemandPrediction("test")
    test_error = test_problem.evaluate(best_pos)
    print("Test error of best solution found while training: {}".format(test_error))

if __name__ == '__main__':
    mp.freeze_support()
    main()
