import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize import minimize
from get_data import get_daily_returns_for_multiple_tickers
# Simulate daily returns for a portfolio of assets
n_assets = 15
n_days = 252

tickers = [
    "AAPL", "MSFT", 
    "GOOGL", "AMZN", "NVDA"
    # , "META", "TSLA", "BRK-B", "JPM", "V",
    # "SPY", "QQQ", "VTI", "ARKK", "DIA", "XOM", "CVX", "BA", "CAT", "GE",
    # "JNJ", "PFE", "UNH", "TSM", "NFLX", "INTC", "AMD", "CRM", "NKE", "WMT"
]
np.random.seed(42)  # for reproducibility


# Fetch daily returns for the specified tickers
returns = get_daily_returns_for_multiple_tickers(tickers, start="2024-12-01", end="2025-01-01")
# print(returns)
# print(returns.shape)
mean_daily_returns = np.mean(returns, axis=0)
# print(mean_daily_returns)
cov_matrix = np.cov(returns, rowvar=False)
print("Size of Covariance Matrix: ",cov_matrix.shape)
def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def random_weights_portfolio(mean_returns, cov_matrix):
    weights = np.random.rand(len(tickers))
    weights /= np.sum(weights)  # Normalize to sum to 1

    # Calculate portfolio return
    portfolio_return = np.dot(weights, mean_returns)

    # Calculate portfolio risk (standard deviation)
    portfolio_risk = np.sqrt(portfolio_variance(weights, cov_matrix))
    return portfolio_return, portfolio_risk, weights

# Generate random portfolios
n_portfolios = 100
portfolios = [random_weights_portfolio(mean_daily_returns, cov_matrix) for _ in range(n_portfolios)]

# Extract returns, risks, and weights from portfolios
frontier_returns = [p[0] for p in portfolios]
frontier_risks = [p[1] for p in portfolios]
frontier_weights = [p[2] for p in portfolios]


# print(frontier_weights)
# Convert to numpy arrays
risks = np.array(frontier_risks)
returns_ = np.array(frontier_returns)

# Sort both based on increasing risk
sorted_indices = np.argsort(risks)
risks = risks[sorted_indices]
returns_ = returns_[sorted_indices]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(risks, returns_, 'r-', linewidth=2, label='Efficient Frontier')
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Expected Return')
plt.title(f'Optimized Markowitz Efficient Frontier for {len(tickers)} Assets')
plt.grid(True)
plt.legend()
plt.show()
