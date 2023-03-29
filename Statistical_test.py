import numpy as np
import multiprocessing as mp
from functools import partial
from DemandPrediction_v2 import DemandPrediction
from scipy.stats import f_oneway
from Main_PSO_v4 import pso
from Main_Genetic import ga

bounds = np.array(DemandPrediction.bounds())

# Define cost function
def cost_function(x):
    training_problem = DemandPrediction("train")
    error = training_problem.evaluate(x)
    return error

def run_pso(params):
    cost_function, bounds, n_particles, n_iterations, c1, c2, w = params
    best_cost, best_position = pso(cost_function, bounds, n_particles, n_iterations, c1, c2, w, verbose=False)
    test_problem = DemandPrediction("test")
    test_error = test_problem.evaluate(best_position)
    return test_error

def run_genetic(params):
    cost_function, bounds, pop_size, num_generations, mutation_rate = params
    best_solution, best_cost = ga(cost_function, bounds, pop_size, num_generations, mutation_rate, verbose=False)
    test_problem = DemandPrediction("test")
    test_error = test_problem.evaluate(best_solution)
    return test_error

def compare_models(num_runs, pso_params, genetic_params):
    pool = mp.Pool(mp.cpu_count())

    pso_results = pool.map(run_pso, [pso_params] * num_runs)
    genetic_results = pool.map(run_genetic, [genetic_params] * num_runs)

    pool.close()
    pool.join()

    f_stat, p_value = f_oneway(pso_results, genetic_results)
    
    print("PSO Results: ", pso_results)
    print("Genetic Algorithm Results: ", genetic_results)
    print("ANOVA F Statistic: ", f_stat)
    print("ANOVA P Value: ", p_value)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    
    num_runs = 30
    pso_params = (cost_function, bounds, 100, 1000, 0.7, 0.7, 0.7)
    genetic_params = (cost_function, bounds, 150, 1500, 0.2)

    compare_models(num_runs, pso_params, genetic_params)
