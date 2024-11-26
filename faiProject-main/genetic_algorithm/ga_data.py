import os
import pandas as pd
import numpy as np


def load_cleaned_data(mean_returns_path, covariance_matrix_path):
    """
    Load processed mean returns and covariance matrix from CSV files.
    """
    # mean returns
    mean_returns = pd.read_csv(mean_returns_path, index_col=0).to_numpy().flatten()

    # covariance matrix
    covariance_matrix = pd.read_csv(covariance_matrix_path, index_col=0).to_numpy()

    # Number of assets
    num_assets = len(mean_returns)

    return mean_returns, covariance_matrix, num_assets

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    mean_returns_path = os.path.join(base_dir, "../clean_data/mean_returns.csv")
    covariance_matrix_path = os.path.join(base_dir, "../clean_data/covariance_matrix.csv")

    # Load the data
    mean_returns, covariance_matrix, num_assets = load_cleaned_data(mean_returns_path, covariance_matrix_path)

    # print results
    print("Mean Returns:", mean_returns)
    print("Covariance Matrix:", covariance_matrix)
    print("Number of Assets:", num_assets)
