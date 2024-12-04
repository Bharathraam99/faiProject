# FAI Project

Our project will focus on portfolio optimization techniques. We aim to evaluate and compare three different algorithms for optimizing a financial stock portfolio, assessing the risk and return. We will implement and analyze Genetic Algorithms, Reinforcement Learning, and the Markowitz Mean-Variance Optimization. These approaches have different methods for balancing both risk and return and will provide a good comparison for finding the optimal technique.

Instructions to run

# Genetic Algorithm

1. Run genetic_algorithm/data_tranformation.py -> Prepares the data (cleaned dataset, covariance matrix, mean returns) from raw data for the genetic algorithm
2. Run genetic_algorithm/main.py -> Runs the genetic algorithm for portfolio optimization
3. genetic_algorithm/processed_data/optimized_portfolio.csv -> Has the end result
4. genetic_algorithm/processed_data/portfolio_allocation.png -> End result visualization

# Reinforcement Learning

1. TODO

# Markowitz Mean-Variance

1. Prepare the investment dataset with CSV formate. The investment goods are not limited, it can be stocks, options, bonds, gold and so on. Each row of the dataset is the date, column represent each individual investment good, each grid represent the final value of the investment good in that date.
2. Run the MarkowitzPortfolioOptimizerMain.py -> you can get weights max sharpe and weights min variance.
3. call MarkowitzPortfolioOptimizer(data).plot_covariance_matrix() can let you get covariance matrix.

# Compare Results

1. Run algoComparator.py to get the comparison of all the three models

# Misc

1. If you want the model to be trained on latest data run newDataGeneration.py
