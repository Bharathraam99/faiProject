import numpy as np
import pandas as pd

# Genetic Algorithm Weights
genetic_weights = pd.read_csv('genetic_algorithm/processed_data/optimized_portfolio.csv')["Weight"]

# Mean-Variance Weights
mean_variance_weights = pd.read_csv('markowitz_mean_variance/markowitz_output.csv')["Weight"]

# Reinforcement Learning Weights
reinforcement_weights = pd.read_csv('Reinforcement learning/RL_portfolio_weights.csv')["Weight"]

# Mean Returns
mean_returns = pd.read_csv('genetic_algorithm/processed_data/mean_returns.csv')["Mean Return"]

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

# Calculate Performance for Each Portfolio
genetic_return, genetic_volatility, genetic_sharpe = portfolio_performance(genetic_weights, mean_returns, covariance_matrix)
rl_return, rl_volatility, rl_sharpe = portfolio_performance(reinforcement_weights, mean_returns, covariance_matrix)
mv_return, mv_volatility, mv_sharpe = portfolio_performance(mean_variance_weights, mean_returns, covariance_matrix)

# Display Results
print("Genetic Algorithm Portfolio:")
print(f"With $1000 invested, the portfolio generates an average daily return of ${genetic_return * 1000:.4f}.")
print(f"The daily volatility (risk) is ±${genetic_volatility * 1000:.4f}, and the Sharpe Ratio (risk-adjusted return) is {genetic_sharpe:.4f}.")

print("\nReinforcement Learning Portfolio:")
print(f"With $1000 invested, the portfolio generates an average daily return of ${rl_return * 1000:.4f}.")
print(f"The daily volatility (risk) is ±${rl_volatility * 1000:.4f}, and the Sharpe Ratio (risk-adjusted return) is {rl_sharpe:.4f}.")

print("\nMean-Variance Portfolio:")
print(f"With $1000 invested, the portfolio generates an average daily return of ${mv_return * 1000:.4f}.")
print(f"The daily volatility (risk) is ±${mv_volatility * 1000:.4f}, and the Sharpe Ratio (risk-adjusted return) is {mv_sharpe:.4f}.")
