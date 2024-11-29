import numpy as np
from crossover import crossover
from mutation import mutate
from selection import select_parents
from fitness import fitness_function


class GeneticAlgorithm:
    def __init__(self, mean_returns, covariance_matrix, population_size, generations, mutation_rate):
        self.mean_returns = mean_returns.values
        self.covariance_matrix = covariance_matrix.values
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.num_assets = len(mean_returns)
        self.population = self.initialize_population()

    def initialize_population(self):
        """
        Initialize a random population of portfolio weights.
        """
        return np.random.dirichlet(np.ones(self.num_assets), size=self.population_size)

    def evolve(self):
        """
        Run the Genetic Algorithm.
        """
        for generation in range(self.generations):
            fitness_scores = np.array([fitness_function(ind, self.mean_returns, self.covariance_matrix) 
                                        for ind in self.population])

            print(f"Generation {generation + 1}: Best Fitness = {max(fitness_scores):.4f}")

            next_population = []

            # Elitism: Keep the best-performing portfolio
            best_individual = self.population[np.argmax(fitness_scores)]
            next_population.append(best_individual)

            while len(next_population) < self.population_size:
                parent1, parent2 = select_parents(self.population, fitness_scores)
                child1, child2 = crossover(parent1, parent2, self.num_assets)
                next_population.append(mutate(child1, self.mutation_rate))
                if len(next_population) < self.population_size:
                    next_population.append(mutate(child2, self.mutation_rate))

            self.population = np.array(next_population)

        best_index = np.argmax([fitness_function(ind, self.mean_returns, self.covariance_matrix) 
                                for ind in self.population])
        return self.population[best_index]
