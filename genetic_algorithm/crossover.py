import numpy as np


def crossover(parent1, parent2, num_assets):
    """
    Perform crossover between two parents.
    """
    crossover_point = np.random.randint(1, num_assets - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2
