import numpy as np


def select_parents(population, fitness_scores):
    """
    Select two parents using roulette wheel selection.
    """
    total_fitness = np.sum(fitness_scores)
    if total_fitness > 0:
        probabilities = fitness_scores / total_fitness
    else:  # All fitness scores are zero; use uniform probabilities
        probabilities = np.ones(len(fitness_scores)) / len(fitness_scores)
    parents_indices = np.random.choice(len(population), size=2, p=probabilities)
    return population[parents_indices]