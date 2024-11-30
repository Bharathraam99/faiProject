import pandas as pd


def load_data(mean_returns_file, covariance_file, daily_returns_file):
    """
    Load all necessary data for the genetic algorithm.
    """
    mean_returns = pd.read_csv(mean_returns_file, index_col=0).squeeze()
    covariance_matrix = pd.read_csv(covariance_file, index_col=0)
    daily_returns = pd.read_csv(daily_returns_file, index_col="Date", parse_dates=True)

    return mean_returns, covariance_matrix, daily_returns
