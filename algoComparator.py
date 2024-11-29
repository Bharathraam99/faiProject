import numpy as np
import pandas as pd

# Genetic Algorithm Weights
genetic_weights = pd.read_csv('genetic_algorithm/processed_data/optimized_portfolio.csv')["Weight"]

# Mean-Variance Weights
mean_variance_weights = np.array([0.0, 0.0, 0.0, 0.05011, 0.0, 0.01629, 0.00834, 
                                   0.0, 0.14477, 0.0, 0.78049, 0.0])

# Mean Returns
mean_returns = np.array([0.002955, 0.002096, 0.001718, 0.004714, 0.001863, 0.000767, 
                          0.032739, 0.002099, 0.004642, 0.002289, 0.007932, 0.001810])

# Covariance Matrix (Load from your file)
covariance_matrix = pd.read_csv("genetic_algorithm/processed_data/covariance_matrix.csv", index_col=0).values

# Risk-Free Rate
risk_free_rate = 0.0001

# Function to calculate portfolio performance
def portfolio_performance(weights, mean_returns, covariance_matrix):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
    return portfolio_return, portfolio_volatility, sharpe_ratio

# Genetic Algorithm Performance
genetic_return, genetic_volatility, genetic_sharpe = portfolio_performance(genetic_weights, mean_returns, covariance_matrix)

# Mean-Variance Performance
mv_return, mv_volatility, mv_sharpe = portfolio_performance(mean_variance_weights, mean_returns, covariance_matrix)

# Display Results
print("Genetic Algorithm Portfolio:")
print(f"Return: {genetic_return:.6f}, Volatility: {genetic_volatility:.6f}, Sharpe Ratio: {genetic_sharpe:.6f}")

print("\nMean-Variance Portfolio:")
print(f"Return: {mv_return:.6f}, Volatility: {mv_volatility:.6f}, Sharpe Ratio: {mv_sharpe:.6f}")
