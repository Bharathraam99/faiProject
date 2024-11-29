import numpy as np


def fitness_function(weights, mean_returns, covariance_matrix):
    """
    Calculate the fitness of a portfolio.
    Fitness = Expected Return / Risk (Sharpe ratio proxy)
    """
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

    if portfolio_risk == 0:
        return 0

    return portfolio_return / portfolio_risk
