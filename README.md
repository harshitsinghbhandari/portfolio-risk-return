# Portfolio Optimization Framework

A comprehensive Python framework for modern portfolio optimization and analysis, implementing multiple optimization strategies including Markowitz Mean-Variance, Risk Parity, and efficient frontier generation.

## Features

### Core Optimization Methods
- **Markowitz Mean-Variance Optimization** - Classic efficient frontier generation
- **Risk Parity Optimization** - Equal risk contribution portfolios
- **Special Portfolio Identification** - Minimum variance, maximum Sharpe ratio, tangency portfolios
- **Short Selling Support** - Toggle between long-only and long-short strategies
- **Multiple Frontier Generation Methods** - Target return or target risk approaches

### Analysis & Visualization
- **Comprehensive Risk-Return Plots** - Compare multiple optimization methods
- **Portfolio Composition Analysis** - Detailed weight and risk contribution breakdowns
- **Performance Metrics** - Sharpe ratio, information ratio, risk contributions
- **Random Portfolio Comparison** - Monte Carlo simulation for benchmarking

### Modular Architecture
- **Object-Oriented Design** - Clean inheritance structure with abstract base classes
- **Extensible Framework** - Easy to add new optimization methods
- **Separation of Concerns** - Optimization logic isolated from analysis and visualization
- **Reusable Components** - Import optimizers into other projects

## 📋 Requirements

- Python 3.7+
- See `requirements.txt` for complete dependency list

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/harshitsinghbhandari/portfolio-risk-return.git
   cd portfolio-risk-return
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure data module availability**
   - Make sure `get_data.py` is in your project directory
   - This module should contain the `get_daily_returns_for_multiple_tickers()` function

## Quick Start

### Basic Usage

```python
from portfolio_optimizer import generate_efficient_frontier, MarkowitzOptimizer
import numpy as np

# Your return data and covariance matrix
mean_returns = np.array([0.001, 0.0008, 0.0012])  # Daily returns
cov_matrix = np.array([[0.0001, 0.00005, 0.00002],
                       [0.00005, 0.0002, 0.00001],
                       [0.00002, 0.00001, 0.00015]])

# Generate efficient frontier
returns, risks, weights = generate_efficient_frontier(
    mean_returns, cov_matrix, n_points=50, allow_short_selling=False
)

print(f"Generated {len(returns)} efficient portfolios")
```

### Advanced Usage

```python
# Create optimizer instance
optimizer = MarkowitzOptimizer(mean_returns, cov_matrix, 
                              asset_names=['AAPL', 'MSFT', 'GOOGL'])

# Find special portfolios
min_var_weights, min_var_return, min_var_risk = optimizer.find_minimum_variance_portfolio()
tang_weights, tang_return, tang_risk = optimizer.find_tangency_portfolio(risk_free_rate=0.02)

# Risk parity optimization
from portfolio_optimizer import RiskParityOptimizer
rp_optimizer = RiskParityOptimizer(mean_returns, cov_matrix)
rp_weights, rp_return, rp_risk = rp_optimizer.optimize()
```

### Run Complete Analysis

```bash
python main.py
```

This will:
- Fetch market data for configured tickers
- Generate multiple efficient frontiers
- Identify special portfolios
- Create comprehensive visualizations
- Output detailed portfolio analysis

## Project Structure

```
portfolio-optimization/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── main.py                  # Main analysis script
├── portfolio_optimizer.py   # Core optimization module
└── get_data.py             # Data fetching module 
```

## Core Components

### PortfolioOptimizer (Abstract Base Class)
```python
class PortfolioOptimizer(ABC):
    def __init__(self, mean_returns, cov_matrix, asset_names=None)
    def portfolio_return(self, weights)      # Calculate expected return
    def portfolio_variance(self, weights)    # Calculate portfolio variance
    def portfolio_risk(self, weights)        # Calculate portfolio risk (std dev)
    @abstractmethod
    def optimize(self, **kwargs)             # Implement in subclasses
```

### MarkowitzOptimizer
```python
class MarkowitzOptimizer(PortfolioOptimizer):
    def minimize_risk_for_target_return(self, target_return)
    def maximize_return_for_target_risk(self, target_risk)
    def find_minimum_variance_portfolio(self)
    def find_tangency_portfolio(self, risk_free_rate=0.0)
```

### RiskParityOptimizer
```python
class RiskParityOptimizer(PortfolioOptimizer):
    def risk_contributions(self, weights)
    def optimize(self, target_risk_contrib=None)
```

## Example Output

### Console Output
```
🚀 Portfolio Optimization Analysis
==================================================
Fetching market data...
Data loaded: 2 assets, 31 days
Covariance matrix size: (2, 2)

Method 1: Classic Markowitz Optimization
Generated 30 efficient portfolios

Method 2: Markowitz with Short Selling
Generated 30 efficient portfolios (with short selling)

Method 3: Special Portfolio Analysis
Minimum Variance Portfolio: Return=0.0008, Risk=0.0156
Tangency Portfolio: Return=0.0012, Risk=0.0189, Sharpe=0.0254

Minimum Variance Portfolio:
---------------------------
Expected Return: 0.0008
Risk (Std Dev):  0.0156
Sharpe Ratio:    0.0254
Asset Allocation:
  AAPL: 0.643 (64.3%)
  MSFT: 0.357 (35.7%)
```

### Visualization
The framework generates comprehensive plots showing:
- **Efficient Frontiers** - Multiple optimization methods
- **Random Portfolios** - Monte Carlo baseline
- **Individual Assets** - Single-asset positions
- **Special Portfolios** - Key portfolio markers

## Configuration

### Asset Selection
Modify the `tickers` list in `main.py`:
```python
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    # Add more tickers as needed
]
```

### Time Period
Adjust data fetching parameters:
```python
returns = get_daily_returns_for_multiple_tickers(
    tickers, 
    start="2024-01-01", 
    end="2024-12-31"
)
```

### Optimization Parameters
```python
# Efficient frontier generation
efficient_returns, efficient_risks, efficient_weights = generate_efficient_frontier(
    mean_daily_returns, 
    cov_matrix, 
    n_points=50,                    # Number of frontier points
    allow_short_selling=False,      # Enable/disable short selling
    method='target_return'          # 'target_return' or 'target_risk'
)
```

## 🔬 Extending the Framework

### Adding New Optimization Methods

1. **Create a new optimizer class**:
```python
class MyCustomOptimizer(PortfolioOptimizer):
    def optimize(self, **kwargs):
        # Implement your optimization logic
        return weights, portfolio_return, portfolio_risk
```

2. **Import and use in main.py**:
```python
from portfolio_optimizer import MyCustomOptimizer

custom_optimizer = MyCustomOptimizer(mean_returns, cov_matrix)
custom_weights, custom_return, custom_risk = custom_optimizer.optimize()
```

### Adding New Risk Measures

Extend the base class with additional risk calculations:
```python
def portfolio_cvar(self, weights, confidence_level=0.05):
    """Calculate Conditional Value at Risk"""
    # Your CVaR implementation
    pass
```

## Mathematical Background

### Markowitz Mean-Variance Optimization
The framework solves the quadratic programming problem:

**Minimize**: `w^T Σ w` (portfolio variance)  
**Subject to**: 
- `Σw = 1` (weights sum to 1)
- `w^T μ = target_return` (achieve target return)
- `0 ≤ w ≤ 1` (long-only) or `-1 ≤ w ≤ 1` (allow short selling)

### Risk Parity
Optimizes for equal risk contributions:
`RC_i = w_i × (∂σ/∂w_i)/σ = 1/n` for all assets

### Sharpe Ratio Maximization
Finds the tangency portfolio: `max[(μ^T w - r_f)/√(w^T Σ w)]`

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-optimizer`)
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## Performance Notes

- **Optimization Speed**: SLSQP method typically converges in 10-50 iterations
- **Memory Usage**: Scales O(n²) with number of assets due to covariance matrix
- **Numerical Stability**: Uses regularization for near-singular covariance matrices

## Important Notes

- **Data Quality**: Ensure your return data is properly cleaned and adjusted for corporate actions
- **Stationarity**: The framework assumes return distributions are stationary
- **Transaction Costs**: Current implementation doesn't include transaction costs
- **Market Frictions**: Assumes perfect liquidity and divisibility of assets

## Troubleshooting

### Common Issues

**Optimization fails to converge**:
- Check if covariance matrix is positive semi-definite
- Reduce target return range
- Try different initial weights

**Empty efficient frontier**:
- Verify return and covariance matrix dimensions match
- Check for missing or invalid data
- Ensure feasible optimization constraints

**Memory issues with large portfolios**:
- Consider using sparse matrices for large covariance matrices
- Implement block-wise optimization for 100+ assets

## License
# Licensed under the MIT License

## 🙏 Acknowledgments

- Built on top of NumPy and SciPy optimization libraries
- Inspired by Markowitz's Modern Portfolio Theory
- Mathematical foundations from Merton's continuous-time finance

---

**Happy Optimizing!**