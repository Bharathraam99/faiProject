import numpy as np


def portfolio_return(weights, mean_returns):
    """
    Calculate the expected portfolio return.
    """
    return np.dot(weights, mean_returns)


def portfolio_risk(weights, covariance_matrix):
    """
    Calculate the portfolio risk (standard deviation).
    """
    return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
