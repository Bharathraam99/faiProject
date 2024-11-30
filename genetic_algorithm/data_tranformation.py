import os
import pandas as pd
import numpy as np


def merge_csv_files(input_folder, output_file):
    """
    Merges all CSV files in a folder on the 'Date' column and saves the merged dataset.

    Args:
        input_folder (str): Path to the folder containing raw CSV files.
        output_file (str): Path to save the merged dataset.

    Returns:
        pd.DataFrame: The merged dataset.
    """
    merged_data = None

    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            file_path = os.path.join(input_folder, file)
            stock_data = pd.read_csv(file_path)

            if 'Date' not in stock_data.columns or 'Close' not in stock_data.columns:
                print(f"Skipping {file}: Missing required columns")
                continue

            stock_data = stock_data[['Date', 'Close']]
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            stock_name = file.split('.')[0]
            stock_data = stock_data.rename(columns={'Close': stock_name})

            # Merge data on Date column
            if merged_data is None:
                merged_data = stock_data
            else:
                merged_data = pd.merge(merged_data, stock_data, on='Date', how='outer')

    merged_data.to_csv(output_file, index=False)
    print(f"Merged data saved to {output_file}")

    return merged_data


def calculate_daily_returns(cleaned_data, output_folder):
    """
    Calculates daily returns from the cleaned dataset and saves the results.

    Args:
        cleaned_data (pd.DataFrame): The cleaned dataset with aligned dates and stock prices.
        output_folder (str): Path to save the daily returns file.

    Returns:
        pd.DataFrame: DataFrame of daily returns.
    """
    daily_returns = cleaned_data.set_index('Date').pct_change().dropna()
    
    daily_returns_file = os.path.join(output_folder, "daily_returns.csv")
    daily_returns.to_csv(daily_returns_file)
    print(f"Daily returns saved to {daily_returns_file}")

    return daily_returns


def calculate_mean_returns(daily_returns, output_folder):
    """
    Calculates mean returns for each stock and saves the results.

    Args:
        daily_returns (pd.DataFrame): DataFrame of daily returns.
        output_folder (str): Path to save the mean returns file.

    Returns:
        pd.Series: Series of mean returns for each stock.
    """
    mean_returns = daily_returns.mean()
    
    # Save the mean returns
    mean_returns_file = os.path.join(output_folder, "mean_returns.csv")
    mean_returns.to_csv(mean_returns_file, header=["Mean Return"])
    print(f"Mean returns saved to {mean_returns_file}")

    return mean_returns


def calculate_covariance_matrix(daily_returns, output_folder):
    """
    Calculates the covariance matrix of daily returns and saves the results.

    Args:
        daily_returns (pd.DataFrame): DataFrame of daily returns.
        output_folder (str): Path to save the covariance matrix file.

    Returns:
        pd.DataFrame: Covariance matrix of daily returns.
    """
    covariance_matrix = daily_returns.cov()
    
    # Save the covariance matrix
    covariance_matrix_file = os.path.join(output_folder, "covariance_matrix.csv")
    covariance_matrix.to_csv(covariance_matrix_file)
    print(f"Covariance matrix saved to {covariance_matrix_file}")

    return covariance_matrix


def save_risk_free_rate(risk_free_rate, output_folder):
    """
    Saves the risk-free rate to a text file.

    Args:
        risk_free_rate (float): The risk-free rate.
        output_folder (str): Path to save the risk-free rate file.
    """
    risk_free_rate_file = os.path.join(output_folder, "risk_free_rate.txt")
    with open(risk_free_rate_file, "w") as f:
        f.write(str(risk_free_rate))
    print(f"Risk-free rate saved to {risk_free_rate_file}")


def prepare_genetic_algo_inputs(input_folder, cleaned_file, output_folder, risk_free_rate=0.0001):
    """
    Merges raw data, prepares all necessary inputs for a genetic algorithm, and saves the results.

    Args:
        input_folder (str): Path to the folder containing raw CSV files.
        cleaned_file (str): Path to save the cleaned dataset.
        output_folder (str): Path to save the prepared input files.
        risk_free_rate (float): The daily risk-free rate (default: 0.0001).

    Returns:
        dict: Paths to the saved input files.
    """
    merged_data = merge_csv_files(input_folder, cleaned_file)

    merged_data.dropna(inplace=True)

    numeric_cols = merged_data.select_dtypes(include=['number']).columns
    merged_data[numeric_cols] = merged_data[numeric_cols].abs()

    merged_data.to_csv(cleaned_file, index=False)
    print(f"Cleaned data saved to {cleaned_file}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    daily_returns = calculate_daily_returns(merged_data, output_folder)

    mean_returns = calculate_mean_returns(daily_returns, output_folder)

    covariance_matrix = calculate_covariance_matrix(daily_returns, output_folder)

    save_risk_free_rate(risk_free_rate, output_folder)

    return {
        "daily_returns_file": os.path.join(output_folder, "daily_returns.csv"),
        "mean_returns_file": os.path.join(output_folder, "mean_returns.csv"),
        "covariance_matrix_file": os.path.join(output_folder, "covariance_matrix.csv"),
        "risk_free_rate_file": os.path.join(output_folder, "risk_free_rate.txt"),
    }


if __name__ == "__main__":
    input_folder = "new_dataset/historical_data"
    cleaned_file = "genetic_algorithm/processed_data/cleaned_data.csv"
    output_folder = "genetic_algorithm/processed_data"

    input_files = prepare_genetic_algo_inputs(input_folder, cleaned_file, output_folder)

    print("\nSaved Input Files:")
    for key, file_path in input_files.items():
        print(f"{key}: {file_path}")
