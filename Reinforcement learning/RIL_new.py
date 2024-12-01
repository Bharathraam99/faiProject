import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

print("Current Working Directory:", os.getcwd())

def get_data_path():
    """Returns the path to the directory containing the CSV files."""
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the .py file
    download_path = os.path.join(current_dir, '..', 'new_dataset', 'historical_data')
    return os.path.normpath(download_path)


def load_and_merge_data(data_path):
    """Loads and merges CSV files from the specified directory."""
    merged_data = pd.DataFrame()
    
    # Loop through each file in the directory
    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            company_name = file.replace(".csv", "")
            file_path = os.path.join(data_path, file)
            df = pd.read_csv(file_path)
            df["Company"] = company_name
            merged_data = pd.concat([merged_data, df], ignore_index=True)
    
    return merged_data


def save_merged_data(merged_data, output_filename="merged_stocks_data.csv"):
    """Saves the merged DataFrame to a CSV file."""
    current_dir = os.getcwd()
    output_path = os.path.join(current_dir, '..', output_filename)
    output_path = os.path.normpath(output_path)
    merged_data.to_csv(output_path, index=False)
    print(f"Data merged successfully into {output_path}.")


# Q-Learning Functions
def get_state(merged_data, t, window=5):
    """Extracts the state as a tuple of the last `window` days of data."""
    prices = merged_data.iloc[max(0, t-window+1):t+1][["Open", "High", "Low", "Close", "Volume"]]
    return tuple(prices.values.flatten())


def get_reward(action, current_price, shares_held, balance, prev_value):
    """Calculates the reward and updates the portfolio value."""
    current_value = balance + shares_held * current_price
    reward = current_value - prev_value

    # Add bonuses for specific actions
    if action == "Sell" and shares_held > 0:
        reward += shares_held * current_price * 0.1  # Incentivize profitable selling
    elif action == "Buy" and balance >= current_price:
        reward += 0.05 * current_price  # Incentivize buying

    return reward, current_value


def train_q_learning(merged_data, unique_companies, actions, alpha, gamma, epsilon, epsilon_decay, initial_balance, episodes=100):
    """Trains a Q-learning agent on the merged stock data."""
    q_table = {}
    portfolio = {company: 0 for company in unique_companies}

    for episode in range(1, episodes + 1):
        balance = initial_balance
        portfolio = {company: 0 for company in unique_companies}
        prev_value = balance
        total_reward = 0

        for t in range(len(merged_data) - 1):
            state = get_state(merged_data, t)
            company = merged_data.loc[t, "Company"]

            # Initialize Q-values for unseen states
            if state not in q_table:
                q_table[state] = [0] * len(actions)

            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action_index = random.randint(0, len(actions) - 1)  # Explore
            else:
                action_index = np.argmax(q_table[state])  # Exploit

            action = actions[action_index]
            current_price = merged_data.loc[t, "Close"]
            shares_held = portfolio[company]

            # Calculate reward and update previous value
            reward, prev_value = get_reward(action, current_price, shares_held, balance, prev_value)

            # Update portfolio and balance
            if action == "Buy" and balance >= current_price:
                balance -= current_price
                portfolio[company] += 1
            elif action == "Sell" and shares_held > 0:
                balance += shares_held * current_price
                portfolio[company] = 0

            total_reward += reward

            # Next state
            next_state = get_state(merged_data, t + 1)
            if next_state not in q_table:
                q_table[next_state] = [0] * len(actions)

            # Q-learning update (Bellman equation)
            old_q_value = q_table[state][action_index]
            next_max_q = max(q_table[next_state])
            q_table[state][action_index] = old_q_value + alpha * (reward + gamma * next_max_q - old_q_value)

        # Decay epsilon
        epsilon = max(epsilon * epsilon_decay, 0.01)

    print("Training complete.")
    return q_table, portfolio, balance


# Visualization
def plot_portfolio_allocation(portfolio, merged_data):
    """Plots the final portfolio allocation as a pie chart."""
    final_portfolio_value = {
        company: shares * merged_data[merged_data['Company'] == company].iloc[-1]['Close']
        for company, shares in portfolio.items() if shares > 0
    }

    labels = list(final_portfolio_value.keys())
    values = list(final_portfolio_value.values())

    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=labels, autopct='%1.2f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title("Portfolio Allocation by Company (Excluding Cash)")
    plt.show()


# Main Execution
if __name__ == "__main__":
    # Parameters
    actions = ["Buy", "Sell", "Hold"]
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 1.0  # Exploration rate
    epsilon_decay = 0.995
    initial_balance = 10000

    # Load and process data
    data_path = get_data_path()
    merged_data = load_and_merge_data(data_path)
    save_merged_data(merged_data)

    # Extract unique companies
    unique_companies = merged_data["Company"].unique()

    # Train Q-learning agent
    q_table, portfolio, balance = train_q_learning(
        merged_data, unique_companies, actions, alpha, gamma, epsilon, epsilon_decay, initial_balance
    )

    # Plot portfolio allocation
    plot_portfolio_allocation(portfolio, merged_data)
