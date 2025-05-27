import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# Simulate daily returns for a portfolio of assets
n_assets = 15
n_days = 252

np.random.seed(42)  # for reproducibility
mean_returns = np.random.uniform(0.001, 0.02, n_assets)
std_devs = np.random.uniform(0.005, 0.03, n_assets)

# Simulate daily returns
returns = np.stack([
    np.random.normal(loc=mean_returns[i], scale=std_devs[i], size=n_days)
    for i in range(n_assets)
], axis=1)

mean_daily_returns = np.mean(returns, axis=0)
cov_matrix = np.cov(returns, rowvar=False)

# Optimization function
def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

# Constraints and bounds
def optimize_portfolio(target_return, mean_returns, cov_matrix):
    n = len(mean_returns)
    init_guess = np.ones(n) / n
    bounds = tuple((0, 1) for _ in range(n))

    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # sum of weights = 1
        {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target_return}  # target return
    )

    result = minimize(portfolio_variance, init_guess, args=(cov_matrix,),
                      method='SLSQP', bounds=bounds, constraints=constraints)

    return result

# Build frontier
target_returns = np.linspace(min(mean_daily_returns), max(mean_daily_returns), 100)
frontier_returns = []
frontier_risks = []
frontier_weights = []

for target in target_returns:
    result = optimize_portfolio(target, mean_daily_returns, cov_matrix)
    if result.success:
        risk = np.sqrt(result.fun)
        frontier_returns.append(target)
        frontier_risks.append(risk)
        frontier_weights.append(result.x)
    else:
        # Optimization failed — skip this point
        continue

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(frontier_risks, frontier_returns, color='red', linewidth=2, label='Efficient Frontier')
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Expected Return')
plt.title(f'Optimized Markowitz Efficient Frontier for {n_assets} Assets')
plt.grid(True)
plt.legend()
plt.show()
