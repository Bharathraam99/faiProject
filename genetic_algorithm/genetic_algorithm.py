from portfolio import Portfolio
from fitness import FitnessEvaluator
from selection import Selection
from crossover import Crossover
from mutation import Mutation

class GeneticAlgorithm:
    def __init__(self, returns, covariance, num_assets, population_size=50, num_generations=100, mutation_rate=0.01, crossover_rate=0.7):
        self.returns = returns
        self.covariance = covariance
        self.num_assets = num_assets
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.population = [Portfolio.random_portfolio(num_assets) for _ in range(population_size)]
        self.fitness_evaluator = FitnessEvaluator(returns, covariance)

    def run(self):
        return self.population[0]
        # Initialize the population with random portfolios

        # For each generation:
        #     1. Evaluate Fitness:
        #         - For each portfolio in the population:
        #             - Calculate its fitness using the fitness function - TODO
            
        #     2. Selection:
        #         - Select pairs of portfolios (parents) based on their fitness scores - TODO
        #         - Higher fitness portfolios have a higher chance of being selected
            
        #     3. Crossover:
        #         - For each selected pair of parents:
        #             - Perform crossover to create new offspring (child portfolios) - TODO
        #             - Add offspring to the new population
            
        #     4. Mutation:
        #         - For each portfolio in the new population:
        #             - Apply mutation with a certain probability - TODO
        #             - Mutation randomly adjusts some portfolio weights to maintain diversity
            
        #     5. Update Population:
        #         - Replace the old population with the new population of offspring - TODO
