

import numpy as np
from scipy.optimize import minimize
from abc import ABC, abstractmethod


class PortfolioOptimizer(ABC):
    """Abstract base class for portfolio optimizers"""
    
    def __init__(self, mean_returns, cov_matrix, asset_names=None):
        self.mean_returns = np.array(mean_returns)
        self.cov_matrix = np.array(cov_matrix)
        self.n_assets = len(mean_returns)
        self.asset_names = asset_names or [f"Asset_{i}" for i in range(self.n_assets)]
    
    @abstractmethod
    def optimize(self, **kwargs):
        pass
    
    def portfolio_return(self, weights):
        return np.dot(weights, self.mean_returns)
    
    def portfolio_variance(self, weights):
        return np.dot(weights.T, np.dot(self.cov_matrix, weights))
    
    def portfolio_risk(self, weights):
        return np.sqrt(self.portfolio_variance(weights))


class MarkowitzOptimizer(PortfolioOptimizer):
    """Classic Markowitz mean-variance optimizer"""
    
    def __init__(self, mean_returns, cov_matrix, asset_names=None, allow_short_selling=False):
        super().__init__(mean_returns, cov_matrix, asset_names)
        self.allow_short_selling = allow_short_selling
    
    def minimize_risk_for_target_return(self, target_return):
        """Find minimum risk portfolio for a given target return"""
        
        def objective(weights):
            return self.portfolio_variance(weights)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # weights sum to 1
            {'type': 'eq', 'fun': lambda w: self.portfolio_return(w) - target_return}
        ]
        
        # Bounds
        if self.allow_short_selling:
            bounds = tuple((-1, 1) for _ in range(self.n_assets))
        else:
            bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Initial guess
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x, self.portfolio_risk(result.x)
        else:
            return None, None
    
    def maximize_return_for_target_risk(self, target_risk):
        """Find maximum return portfolio for a given risk level"""
        
        def objective(weights):
            return -self.portfolio_return(weights)  # Negative for maximization
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: self.portfolio_risk(w) - target_risk}
        ]
        
        if self.allow_short_selling:
            bounds = tuple((-1, 1) for _ in range(self.n_assets))
        else:
            bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x, self.portfolio_return(result.x)
        else:
            return None, None
    
    def find_minimum_variance_portfolio(self):
        """Find the global minimum variance portfolio"""
        
        def objective(weights):
            return self.portfolio_variance(weights)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        if self.allow_short_selling:
            bounds = tuple((-1, 1) for _ in range(self.n_assets))
        else:
            bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x, self.portfolio_return(result.x), self.portfolio_risk(result.x)
        else:
            return None, None, None
    
    def find_tangency_portfolio(self, risk_free_rate=0.0):
        """Find the tangency (maximum Sharpe ratio) portfolio"""
        
        def objective(weights):
            portfolio_ret = self.portfolio_return(weights)
            portfolio_vol = self.portfolio_risk(weights)
            if portfolio_vol == 0:
                return -np.inf
            return -(portfolio_ret - risk_free_rate) / portfolio_vol  # Negative Sharpe ratio
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        if self.allow_short_selling:
            bounds = tuple((-1, 1) for _ in range(self.n_assets))
        else:
            bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x, self.portfolio_return(result.x), self.portfolio_risk(result.x)
        else:
            return None, None, None
    
    def optimize(self, method='efficient_frontier', **kwargs):
        """Main optimization method"""
        if method == 'efficient_frontier':
            return self.generate_efficient_frontier(**kwargs)
        elif method == 'min_variance':
            return self.find_minimum_variance_portfolio()
        elif method == 'tangency':
            return self.find_tangency_portfolio(**kwargs)
        else:
            raise ValueError(f"Unknown optimization method: {method}")


class RiskParityOptimizer(PortfolioOptimizer):
    """Risk Parity optimizer - equalizes risk contributions"""
    
    def risk_contributions(self, weights):
        """Calculate risk contributions of each asset"""
        portfolio_vol = self.portfolio_risk(weights)
        marginal_contrib = np.dot(self.cov_matrix, weights)
        risk_contrib = weights * marginal_contrib / portfolio_vol
        return risk_contrib
    
    def optimize(self, target_risk_contrib=None):
        """Optimize for risk parity"""
        if target_risk_contrib is None:
            target_risk_contrib = np.array([1/self.n_assets] * self.n_assets)
        
        def objective(weights):
            risk_contrib = self.risk_contributions(weights)
            # Minimize sum of squared deviations from target risk contributions
            return np.sum((risk_contrib - target_risk_contrib) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0.001, 1) for _ in range(self.n_assets))  # Small lower bound to avoid division by zero
        
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x, self.portfolio_return(result.x), self.portfolio_risk(result.x)
        else:
            return None, None, None


def generate_efficient_frontier(mean_returns, cov_matrix, n_points=50, 
                              allow_short_selling=False, method='target_return'):
    """
    Generate efficient frontier using various methods
    
    Parameters:
    -----------
    mean_returns : array-like
        Expected returns for each asset
    cov_matrix : array-like
        Covariance matrix of asset returns
    n_points : int
        Number of points to generate on the frontier
    allow_short_selling : bool
        Whether to allow negative weights
    method : str
        'target_return' or 'target_risk'
    
    Returns:
    --------
    tuple: (frontier_returns, frontier_risks, frontier_weights)
    """
    
    optimizer = MarkowitzOptimizer(mean_returns, cov_matrix, 
                                  allow_short_selling=allow_short_selling)
    
    # Find minimum variance portfolio to determine feasible range
    min_var_weights, min_return, min_risk = optimizer.find_minimum_variance_portfolio()
    
    if min_var_weights is None:
        raise ValueError("Could not find minimum variance portfolio")
    
    # Determine maximum return (100% in best performing asset if no short selling)
    if allow_short_selling:
        max_return = np.max(mean_returns) * 2  # Allow for some leverage
    else:
        max_return = np.max(mean_returns)
    
    frontier_returns = []
    frontier_risks = []
    frontier_weights = []
    
    if method == 'target_return':
        # Generate points by targeting different return levels
        target_returns = np.linspace(min_return, max_return * 0.95, n_points)
        
        for target_ret in target_returns:
            weights, risk = optimizer.minimize_risk_for_target_return(target_ret)
            if weights is not None:
                frontier_returns.append(target_ret)
                frontier_risks.append(risk)
                frontier_weights.append(weights)
    
    elif method == 'target_risk':
        # Generate points by targeting different risk levels
        max_risk = min_risk * 3  # Reasonable upper bound
        target_risks = np.linspace(min_risk, max_risk, n_points)
        
        for target_risk in target_risks:
            weights, ret = optimizer.maximize_return_for_target_risk(target_risk)
            if weights is not None:
                frontier_returns.append(ret)
                frontier_risks.append(target_risk)
                frontier_weights.append(weights)
    
    return np.array(frontier_returns), np.array(frontier_risks), frontier_weights


def generate_random_portfolios(mean_returns, cov_matrix, n_portfolios=1000, 
                             allow_short_selling=False):
    """Generate random portfolios for comparison"""
    n_assets = len(mean_returns)
    random_returns = []
    random_risks = []
    
    for _ in range(n_portfolios):
        if allow_short_selling:
            # Allow negative weights
            weights = np.random.randn(n_assets)
            weights = weights / np.sum(np.abs(weights))  # Normalize by sum of absolute values
        else:
            # Only positive weights
            weights = np.random.rand(n_assets)
            weights /= np.sum(weights)
        
        # Calculate portfolio metrics
        port_return = np.dot(weights, mean_returns)
        port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        random_returns.append(port_return)
        random_risks.append(port_risk)
    
    return np.array(random_returns), np.array(random_risks)


def calculate_portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate=1.0):
    """Calculate comprehensive portfolio metrics"""
    weights = np.array(weights)
    mean_returns = np.array(mean_returns)
    
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_risk = np.sqrt(portfolio_variance)
    
    # Sharpe ratio
    if portfolio_risk > 0:
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
    else:
        sharpe_ratio = 0
    
    # Information ratio (assuming benchmark return is risk-free rate)
    information_ratio = sharpe_ratio  # Simplified
    
    return {
        'return': portfolio_return,
        'risk': portfolio_risk,
        'variance': portfolio_variance,
        'sharpe_ratio': sharpe_ratio,
        'information_ratio': information_ratio,
        'weights': weights
    }