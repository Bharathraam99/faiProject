import numpy as np


def select_parents(population, fitness_scores):
    """
    Select two parents using roulette wheel selection.
    """
    total_fitness = np.sum(fitness_scores)
    probabilities = fitness_scores / total_fitness if total_fitness > 0 else np.ones(len(fitness_scores)) / len(fitness_scores)
    parents_indices = np.random.choice(len(population), size=2, p=probabilities)
    return population[parents_indices]
