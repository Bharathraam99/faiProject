import numpy as np


def mutate(individual, mutation_rate):
    """
    Mutate an individual portfolio by slightly altering weights.
    """
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] += np.random.uniform(-0.1, 0.1)
    individual = np.clip(individual, 0, None)
    return individual / np.sum(individual)
