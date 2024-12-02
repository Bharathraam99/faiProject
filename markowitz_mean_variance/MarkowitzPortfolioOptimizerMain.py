import pandas as pd
from MarkowitzPortfolioOptimizer import MarkowitzPortfolioOptimizer 


file_path = 'markowitz_mean_variance/data/stock_table2.csv'
stock_data = pd.read_csv(file_path)

stock_data.set_index('Date', inplace=True) 
stock_data = stock_data.apply(pd.to_numeric, errors='coerce')  
stock_data = stock_data.dropna() 

optimizer = MarkowitzPortfolioOptimizer(stock_data)

weights_max_sharpe = optimizer.optimize(method='max_sharpe')
print("Maximum Sharpe Ratio Portfolio Weights:")
print(optimizer.display_results())

weights_min_variance = optimizer.optimize(method='min_variance')
print("Minimum Variance Portfolio Weights:")
print(optimizer.display_results())
