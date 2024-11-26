from genetic_algorithm import GeneticAlgorithm


POPULATION_SIZE = None
MUTATION_RATE = None
CROSSOVER_RATE = None
GENERATIONS = None

num_assets = None
returns = None
covariance = None #Need to data prep and get the covariance matrix to feed to the genetic algorithm

# Initialize and run the genetic algorithm
ga = GeneticAlgorithm(returns, covariance, num_assets, POPULATION_SIZE, GENERATIONS, MUTATION_RATE, CROSSOVER_RATE)
best_portfolio = ga.run()
print("Best Portfolio Weights:", best_portfolio.weights)
