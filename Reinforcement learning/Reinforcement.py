import os
import random
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import ta

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Constants
ACTIONS = ["Buy_1", "Buy_5", "Sell_1", "Sell_5", "Hold"]
ALPHA = 0.05
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
INITIAL_BALANCE = 10000
TRANSACTION_COST_PERCENTAGE = 0.001
EPISODES = 100

# Load and preprocess data
def load_and_preprocess_data(base_dir, file_name, company_name):
    data_path = Path(base_dir) / file_name
    if not data_path.exists():
        raise FileNotFoundError(f"The file {data_path} does not exist.")
    
    data = pd.read_csv(data_path)
    company_data = data[data['Company'] == company_name].reset_index(drop=True)
    
    # Add technical indicators
    company_data['MA10'] = company_data['Close'].rolling(window=10).mean()
    company_data['MA50'] = company_data['Close'].rolling(window=50).mean()
    company_data['RSI'] = compute_rsi(company_data['Close'], window=14)
    company_data['RSI_Category'] = company_data['RSI'].apply(discretize_rsi)
    
    # Drop NaN rows and normalize features
    company_data = company_data.dropna().reset_index(drop=True)
    scaler = StandardScaler()
    company_data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA50', 'RSI']] = scaler.fit_transform(
        company_data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA50', 'RSI']]
    )
    return company_data

def compute_rsi(series, window=14):
    return ta.momentum.RSIIndicator(series, window=window).rsi()

def discretize_rsi(rsi_value):
    if rsi_value < 30:
        return 'Oversold'
    elif 30 <= rsi_value < 70:
        return 'Neutral'
    else:
        return 'Overbought'

# Q-learning state representation
def get_state(data, t):
    return tuple(data.iloc[t][["Close", "MA10", "MA50", "RSI"]])

# Reward function
def get_reward(action, current_price, shares_held, balance, previous_portfolio_value):
    if action == "Buy_1" and balance >= (current_price * (1 + TRANSACTION_COST_PERCENTAGE)):
        balance -= current_price * (1 + TRANSACTION_COST_PERCENTAGE)
        shares_held += 1
    elif action == "Buy_5" and balance >= (current_price * 5 * (1 + TRANSACTION_COST_PERCENTAGE)):
        balance -= current_price * 5 * (1 + TRANSACTION_COST_PERCENTAGE)
        shares_held += 5
    elif action == "Sell_1" and shares_held >= 1:
        balance += current_price * (1 - TRANSACTION_COST_PERCENTAGE)
        shares_held -= 1
    elif action == "Sell_5" and shares_held >= 5:
        balance += current_price * 5 * (1 - TRANSACTION_COST_PERCENTAGE)
        shares_held -= 5

    new_portfolio_value = balance + shares_held * current_price
    reward = new_portfolio_value - previous_portfolio_value
    return reward, balance, shares_held

# Train the Q-learning agent
# Pass epsilon as an argument instead of using a global variable
def train_agent(data, epsilon):
    q_table = {}
    episode_rewards = []

    for episode in range(1, EPISODES + 1):
        balance = INITIAL_BALANCE
        shares_held = 0
        total_reward = 0
        actions_taken = []
        
        for t in range(len(data) - 1):
            state = get_state(data, t)
            if state not in q_table:
                q_table[state] = [0] * len(ACTIONS)
            
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action_index = random.randint(0, len(ACTIONS) - 1)
            else:
                action_index = np.argmax(q_table[state])
            
            action = ACTIONS[action_index]
            current_price = data.loc[t, "Close"]
            previous_portfolio_value = balance + shares_held * current_price
            reward, balance, shares_held = get_reward(action, current_price, shares_held, balance, previous_portfolio_value)
            
            actions_taken.append(action)
            total_reward += reward
            
            next_state = get_state(data, t + 1)
            if next_state not in q_table:
                q_table[next_state] = [0] * len(ACTIONS)
            
            old_q_value = q_table[state][action_index]
            next_max_q = max(q_table[next_state])
            q_table[state][action_index] = old_q_value + ALPHA * (reward + GAMMA * next_max_q - old_q_value)
        
        # End of episode: sell remaining shares
        final_price = data.loc[len(data) - 1, "Close"]
        if shares_held > 0:
            balance += final_price * shares_held * (1 - TRANSACTION_COST_PERCENTAGE)
            shares_held = 0
        
        final_portfolio_value = balance
        episode_reward = final_portfolio_value - INITIAL_BALANCE
        episode_rewards.append(episode_reward)
        
        # Update epsilon for exploration-exploitation tradeoff
        if epsilon > EPSILON_MIN:
            epsilon *= EPSILON_DECAY
        
        print(f"Episode {episode}, Reward: {episode_reward:.2f}, Balance: {balance:.2f}, Actions: {pd.Series(actions_taken).value_counts().to_dict()}")

    return q_table, episode_rewards, epsilon


# Plot rewards
def plot_rewards(rewards):
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(rewards) + 1), rewards, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.grid(True)
    plt.show()

# Save Q-table
def save_q_table(q_table, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(q_table, f)
    print(f"Q-table saved at {file_path}.")

def main():
    base_dir = Path.cwd().parent
    file_name = 'merged_stocks_data.csv'
    company_name = 'AAPL'

    data = load_and_preprocess_data(base_dir, file_name, company_name)
    global EPSILON  # You can still keep track of EPSILON globally if needed
    q_table, rewards, EPSILON = train_agent(data, EPSILON)
    plot_rewards(rewards)
    save_q_table(q_table, base_dir / 'q_table.pkl')

if __name__ == "__main__":
    main()

