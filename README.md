# FAI Project - Portfolio Optimization

Our project will focus on portfolio optimization techniques. We aim to evaluate and compare three different algorithms for optimizing a financial stock portfolio, assessing the risk and return. We will implement and analyze Genetic Algorithms, Reinforcement Learning, and the Markowitz Mean-Variance Optimization. These approaches have different methods for balancing both risk and return and will provide a good comparison for finding the optimal technique.

# General Instruction to see the results of each algorithm and run the comparison chart
1. Clone the Repo
2. Genetic Algorithm Results -> genetic_algorithm/processed_data/optimized_portfolio.csv
3. Reinforcement Learning Results -> Reinforcement learning/RL_portfolio_weights.csv
4. Markowitz Mean Variance Results -> markowitz_mean_variance/ouput/markowitz_output.csv
5. To see the comparison chart of all the three algorithms run algoComparator.py

## Below are steps to retrain each of the algorithms from scratch

# Genetic Algorithm

1. Run genetic_algorithm/data_tranformation.py -> Prepares the data (cleaned dataset, covariance matrix, mean returns) from raw data for the genetic algorithm
2. Run genetic_algorithm/main.py -> Runs the genetic algorithm for portfolio optimization
3. genetic_algorithm/processed_data/optimized_portfolio.csv -> Has the end result
4. genetic_algorithm/processed_data/portfolio_allocation.png -> End result visualization

# Reinforcement Learning

1. Run Reinforcement Learning /RIL_py.py -> function to Merge the dataset, dataset is woder with an extra column of company name, (cleaned dataset) 
RIL_new.py -> Visualization Function to generate the pie chart allocating weights to varous stocks and saving the weights allocated to each stock into a csv file for comparison

2. Reinforcement Learning /Reinforcement.py -> Q-learning Agent: Trains an agent to make stock trading decisions (buy, sell, hold) based on historical data.
Reinforcement.py -> Data Processing: Prepares stock data with technical indicators (e.g., RSI, Moving Averages) for the agent to learn from.

# Markowitz Mean-Variance

1. Prepare the investment dataset with CSV formate. The investment goods are not limited, it can be stocks, options, bonds, gold and so on. Each row of the dataset is the date, column represent each individual investment good, each grid represent the final value of the investment good in that date.
2. Run the MarkowitzPortfolioOptimizerMain.py -> you can get weights max sharpe and weights min variance.
3. call MarkowitzPortfolioOptimizer(data).plot_covariance_matrix() can let you get covariance matrix.

# Misc

1. If you want the model to be trained on latest data run newDataGeneration.py and run all the models for new portfolio weights
