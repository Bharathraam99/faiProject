import numpy as np


def fitness_function(weights, mean_returns, covariance_matrix):
    """
    Calculate the fitness of a portfolio.
    Fitness = Expected Return / Risk (Sharpe ratio proxy)
    """
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    diversification_penalty = 1 - np.sum(np.square(weights))  # Higher for diversified portfolios
    return (portfolio_return / portfolio_volatility) * diversification_penalty if portfolio_volatility > 0 else 0.0