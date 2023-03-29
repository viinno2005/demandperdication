import numpy as np
import multiprocessing as mp
from functools import partial
from DemandPrediction_v2 import DemandPrediction
from Main_Genetic import initialize_population, selection, crossover, mutation

# Define cost function
def cost_function(x):
    training_problem = DemandPrediction("train")
    error = training_problem.evaluate(x)
    return error

def ga_with_hyperparameters(hyperparams):
    pop_size, num_generations, mutation_rate = hyperparams

    bounds = np.array(DemandPrediction.bounds())
    num_parents = int(pop_size / 2)
    population = initialize_population(pop_size, bounds)

    for generation in range(num_generations):
        fitness = np.array([cost_function(individual) for individual in population])
        parents = selection(population, fitness.copy(), num_parents)
        offspring = crossover(parents, offspring_size=(pop_size - num_parents, bounds.shape[0]))
        offspring = mutation(offspring, bounds, mutation_rate)
        population[0:parents.shape[0]] = parents
        population[parents.shape[0]:] = offspring

    best_solution = population[np.argmin(fitness)]
    current_best_cost = np.min(fitness)

    return (current_best_cost, best_solution, hyperparams)

# Set ranges for hyperparameter tuning
pop_size_range = [50, 100, 150]
num_generations_range = [500, 1000, 1500]
mutation_rate_range = [0.05, 0.1, 0.2]

# Prepare combinations of hyperparameters
hyperparameter_combinations = [(pop_size, num_generations, mutation_rate) for pop_size in pop_size_range for num_generations in num_generations_range for mutation_rate in mutation_rate_range]

def main():
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(ga_with_hyperparameters, hyperparameter_combinations)

    best_cost = float('inf')
    best_solution = None
    best_hyperparameters = None

    for cost, pos, hyperparams in results:
        print("pop_size: {}, num_generations: {}, mutation_rate: {}, cost: {}".format(hyperparams[0], hyperparams[1], hyperparams[2], cost))
        if cost < best_cost:
            best_cost = cost
            best_solution = pos
            best_hyperparameters = hyperparams

    print("\nBest hyperparameters found:")
    print("pop_size: {}, num_generations: {}, mutation_rate: {}".format(best_hyperparameters[0], best_hyperparameters[1], best_hyperparameters[2]))
    print("Best training error: {}".format(best_cost))

    # Evaluate the best solution on the test dataset
    test_problem = DemandPrediction("test")
    test_error = test_problem.evaluate(best_solution)
    print("Test error of best solution found while training: {}".format(test_error))

if __name__ == '__main__':
    mp.freeze_support()
    main()
