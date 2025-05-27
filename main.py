import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

# 20 days of returns for 2 synthetic assets
n_days = 20
mean_returns = np.array([0.2, 0.03])
std_devs = np.array([0.01, 0.012])

returns_asset_1 = np.random.normal(loc=mean_returns[0], scale=std_devs[0], size=n_days)
returns_asset_2 = np.random.normal(loc=mean_returns[1], scale=std_devs[1], size=n_days)

# Stack into a (20 x 2) matrix
returns = np.column_stack((returns_asset_1, returns_asset_2))

mean_daily_returns = np.mean(returns, axis=0)
cov_matrix = np.cov(returns, rowvar=False)

# Range of weights for asset 1 (asset 2 = 1 - w)
weights = np.linspace(0, 1, 100)
portfolio_returns = []
portfolio_risks = []
portfolio_weights = []

for w in weights:
    w_vec = np.array([w, 1 - w])
    ret = np.dot(w_vec, mean_daily_returns)
    risk = np.sqrt(np.dot(w_vec.T, np.dot(cov_matrix, w_vec)))
    
    portfolio_returns.append(ret)
    portfolio_risks.append(risk)
    portfolio_weights.append(w_vec)
plt.figure(figsize=(10, 6))
plt.plot(portfolio_risks, portfolio_returns, label='Efficient Frontier')
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Expected Return')
plt.title('Markowitz Efficient Frontier for 2 Assets')
plt.grid(True)
plt.legend()
plt.show()
# Print sample portfolios along the frontier
for i in [0, 25, 50, 75, 99]:
    w = portfolio_weights[i]
    print(f"Portfolio {i+1}:")
    print(f"  Asset 1 weight: {w[0]:.2f}, Asset 2 weight: {w[1]:.2f}")
    print(f"  Expected Return: {portfolio_returns[i]:.4f}, Risk: {portfolio_risks[i]:.4f}\n")
