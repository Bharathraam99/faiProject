import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


class MarkowitzPortfolioOptimizer:
    def __init__(self, stock_prices):
        """
        Initialize the Markowitz Portfolio Optimizer.
        :param stock_prices: DataFrame with dates as rows and stock prices as columns.
        """
        if stock_prices.isnull().values.any():
            raise ValueError("Stock price data contains missing values. Please clean your data.")
        if not all([np.issubdtype(dtype, np.number) for dtype in stock_prices.dtypes]):
            raise ValueError("Stock price data contains non-numeric values. Please clean your data.")

        self.stock_prices = stock_prices
        self.returns = self.returns_from_prices(stock_prices)
        self.mean_returns = self.returns.mean()
        self.covariance_matrix = self.sample_cov(self.returns)
        self.weights = None

    def calculate_daily_returns(self):
        """
        Calculate daily returns from stock prices.
        """
        return self.stock_prices.pct_change(fill_method=None).dropna()

    def returns_from_prices(self, prices, log_returns=False):
        """
        Calculate daily returns from price data.
        """
        if log_returns:
            returns = np.log(1 + prices.pct_change(fill_method=None)).dropna(how="all")
        else:
            returns = prices.pct_change(fill_method=None).dropna(how="all")
        return returns

    def check_returns(self, returns):
        """
        Check for NaN or infinite returns.
        """
        if np.any(np.isnan(returns.mask(returns.ffill().isnull(), 0))):
            warnings.warn("NaN returns exist.", UserWarning)
        if np.any(np.isinf(returns)):
            warnings.warn("Infinite returns exist.", UserWarning)

    def mean_historical_return(self, compounding=True, frequency=252):
        """
        Calculate annualized mean historical return.
        """
        returns = self.returns
        self.check_returns(returns)
        if compounding:
            return (1 + returns).prod() ** (frequency / returns.count()) - 1
        else:
            return returns.mean() * frequency

    def is_positive_semidefinite(self, matrix):
        """
        Check if a matrix is positive semidefinite.
        """
        try:
            np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
            return True
        except np.linalg.LinAlgError:
            return False

    def fix_nonpositive_semidefinite(self, matrix, fix_method="spectral"):
        """
        Fix a covariance matrix if it's not positive semidefinite.
        """
        q, V = np.linalg.eigh(matrix)
        if fix_method == "spectral":
            q = np.where(q > 0, q, 0)  
            fixed_matrix = V @ np.diag(q) @ V.T
        elif fix_method == "diag":
            min_eig = np.min(q)
            fixed_matrix = matrix - (1.1 * min_eig) * np.eye(len(matrix))
        else:
            raise NotImplementedError(f"Method {fix_method} not implemented")

        if not self.is_positive_semidefinite(fixed_matrix):
            warnings.warn("Matrix is still not positive semidefinite. Please check your data.", UserWarning)

        if isinstance(matrix, pd.DataFrame):
            tickers = matrix.index
            return pd.DataFrame(fixed_matrix, index=tickers, columns=tickers)
        else:
            return fixed_matrix

    def sample_cov(self, returns, frequency=252, fix_method="spectral"):
        """
        Calculate the annualized sample covariance matrix.
        """
        return self.fix_nonpositive_semidefinite(returns.cov() * frequency, fix_method)
    
    def plot_covariance_matrix(self, cmap='coolwarm', figsize=(10, 8), title='Covariance Matrix Heatmap',vmin=-0.25,vmax=0.25):
        """
        Plot the covariance matrix as a heatmap with enhanced contrast.
        """
        if self.covariance_matrix is None:
            raise ValueError("Covariance matrix has not been calculated.")

        plt.figure(figsize=figsize)
        sns.heatmap(self.covariance_matrix, annot=True, fmt='.2f', cmap=cmap, cbar=True,
                    xticklabels=self.covariance_matrix.columns,
                    yticklabels=self.covariance_matrix.index,
                    square=True, linewidths=0.5, vmin=-0.25,vmax=0.25)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def portfolio_performance(self, weights):
        """
        Calculate portfolio return and volatility.
        """
        port_return = np.dot(weights, self.mean_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
        return port_return, port_volatility

    def optimize(self, risk_free_rate=0.02, method='max_sharpe'):
        """
        Optimize portfolio weights for either maximum Sharpe ratio or minimum variance.
        """
        num_assets = len(self.mean_returns)
        initial_weights = num_assets * [1.0 / num_assets]
        bounds = tuple((0, 1) for _ in range(num_assets))
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        if method == 'max_sharpe':
            def negative_sharpe(weights):
                port_return, port_volatility = self.portfolio_performance(weights)
                return -(port_return - risk_free_rate) / port_volatility

            result = minimize(negative_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        elif method == 'min_variance':
            def portfolio_variance(weights):
                return self.portfolio_performance(weights)[1] ** 2

            result = minimize(portfolio_variance, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        else:
            raise ValueError("Invalid optimization method. Use 'max_sharpe' or 'min_variance'.")

        self.weights = np.round(result.x / result.x.sum(), 5)  # Normalize and round weights
        return self.weights

    def display_results(self):
        """
        Display the optimized portfolio weights.
        """
        if self.weights is None:
            raise ValueError("Optimization has not been run yet.")
        portfolio = pd.DataFrame({
            'Asset': self.stock_prices.columns,
            'Weight': self.weights * 100
        })
        return portfolio
    




# import warnings
# import numpy as np
# import pandas as pd
# from scipy.optimize import minimize

# class MarkowitzPortfolioOptimizer:
#     def __init__(self, stock_prices):
#         """
#         Initialize the Markowitz Portfolio Optimizer.
#         :param stock_prices: DataFrame with dates as rows and stock prices as columns.
#         """
#         self.stock_prices = stock_prices
#         self.returns = self.calculate_daily_returns()
#         self.mean_returns = self.returns.mean()
#         self.covariance_matrix = self.returns.cov()
#         self.weights = None
    
#     def calculate_daily_returns(self):
#         """
#         Calculate daily returns from stock prices.
#         """
#         return self.stock_prices.pct_change().dropna()
    
#     def check_returns(returns):
#         """
#         Check whether the returns are legal.
#         """
#         if np.any(np.isnan(returns.mask(returns.ffill().isnull(), 0))):
#             warnings.warn("NaN returns exist.", UserWarning)
#         if np.any(np.isinf(returns)):
#             warnings.warn("Infinite returns exist.", UserWarning)
    
#     def returns_from_prices(prices, log_returns=False):
#         """
#         Calculate the returns given prices.
#         """
#         if log_returns:
#             returns = np.log(1 + prices.pct_change()).dropna(how="all")
#         else:
#             returns = prices.pct_change().dropna(how="all")
#         return returns
    
#     def mean_historical_return(prices, returns_data=False, compounding=True, frequency=252, log_returns=False):
#         """
#         Calculate annualised mean (daily) historical return from input (daily) asset prices.
#         Use ``compounding`` to toggle between the default geometric mean (CAGR) and the
#         arithmetic mean.
#         """
#         if not isinstance(prices, pd.DataFrame):
#             warnings.warn("Input is not a dataframe", RuntimeWarning)
#             prices = pd.DataFrame(prices)
#         if returns_data:
#             returns = prices
#         else:
#             returns = returns_from_prices(prices, log_returns)
#         check_returns(returns)
#         if compounding:
#             return (1 + returns).prod() ** (frequency / returns.count()) - 1
#         else:
#             return returns.mean() * frequency
    
#     def is_positive_semidefinite(matrix):
#         """
#         Helper function to check if a given matrix is positive semidefinite.
#         Any method that requires inverting the covariance matrix will struggle
#         with a non-positive semidefinite matrix
#         """
#         try:
#             # Significantly more efficient than checking eigenvalues (stackoverflow.com/questions/16266720)
#             np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
#             return True
#         except np.linalg.LinAlgError:
#             return False
    
#     def fix_nonpositive_semidefinite(matrix, fix_method="spectral"):
#         """
#         Check if a covariance matrix is positive semidefinite, and if not, fix it
#         with the chosen method.
#         """

#         # Eigendecomposition
#         q, V = np.linalg.eigh(matrix)

#         if fix_method == "spectral":
#             # Remove negative eigenvalues
#             q = np.where(q > 0, q, 0)
#             # Reconstruct matrix
#             fixed_matrix = V @ np.diag(q) @ V.T
#         elif fix_method == "diag":
#             min_eig = np.min(q)
#             fixed_matrix = matrix - 1.1 * min_eig * np.eye(len(matrix))
#         else:
#             raise NotImplementedError("Method {} not implemented".format(fix_method))

#         if not is_positive_semidefinite(fixed_matrix):  # pragma: no cover
#             warnings.warn(
#                 "Could not fix matrix. Please try a different risk model.", UserWarning
#             )

#         # Rebuild labels if provided
#         if isinstance(matrix, pd.DataFrame):
#             tickers = matrix.index
#             return pd.DataFrame(fixed_matrix, index=tickers, columns=tickers)
#         else:
#             return fixed_matrix

#     def sample_cov(prices, returns_data=False, frequency=252, log_returns=False, **kwargs):
#         """
#         Calculate the annualised sample covariance matrix of (daily) asset returns.
#         """
#         if not isinstance(prices, pd.DataFrame):
#             warnings.warn("Input is not a dataframe", RuntimeWarning)
#             prices = pd.DataFrame(prices)
#         if returns_data:
#             returns = prices
#         else:
#             returns = returns_from_prices(prices, log_returns)
#         return fix_nonpositive_semidefinite(
#             returns.cov() * frequency, kwargs.get("fix_method", "spectral")
#         )