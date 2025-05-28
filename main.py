import numpy as np
import matplotlib.pyplot as plt
from get_data import get_daily_returns_for_multiple_tickers
from portfolio_optimizer import (
    generate_efficient_frontier, 
    generate_random_portfolios,
    MarkowitzOptimizer,
    RiskParityOptimizer,
    calculate_portfolio_metrics
)

# Portfolio configuration
tickers = [
    "AAPL", "MSFT", 
    "GOOGL", "AMZN", "NVDA", 
    "META", "TSLA", "BRK-B", "JPM", "V",
    "SPY", "QQQ", "VTI", "ARKK", "DIA", "XOM", "CVX", "BA", "CAT", "GE",
    # "JNJ", "PFE", "UNH", "TSM", "NFLX", "INTC", "AMD", "CRM", "NKE", "WMT"
]

np.random.seed(42)

def main():
    # Main analysis function
    print("Portfolio Optimization Analysis")
    print("=" * 70)
    
    # Fetch data
    print("Getting market data...")
    returns = get_daily_returns_for_multiple_tickers(tickers, start="2024-12-01", end="2025-01-01")
    mean_daily_returns = np.mean(returns, axis=0)
    cov_matrix = np.cov(returns, rowvar=False)
    
    print(f"Data Added: {len(tickers)} assets, {len(returns)} days")
    print(f"Covariance matrix size: {cov_matrix.shape}")
    
    # Method 1: Classic Markowitz Efficient Frontier
    print("\nMethod 1: Classic Markowitz Optimization")
    efficient_returns, efficient_risks, efficient_weights = generate_efficient_frontier(
        mean_daily_returns, cov_matrix, n_points=30, allow_short_selling=False
    )
    print(f"Generated {len(efficient_returns)} efficient portfolios")
    
    # Method 2: Markowitz with Short Selling
    print("\nMethod 2: Markowitz with Short Selling")
    efficient_returns_short, efficient_risks_short, efficient_weights_short = generate_efficient_frontier(
        mean_daily_returns, cov_matrix, n_points=30, allow_short_selling=True
    )
    print(f"Generated {len(efficient_returns_short)} efficient portfolios (with short selling)")
    
    # Method 3: Find Special Portfolios
    print("\nMethod 3: Special Portfolio Analysis")
    optimizer = MarkowitzOptimizer(mean_daily_returns, cov_matrix, tickers)
    
    # Minimum variance portfolio
    min_var_weights, min_var_return, min_var_risk = optimizer.find_minimum_variance_portfolio()
    print(f"Minimum Variance Portfolio: Return={min_var_return:.4f}, Risk={min_var_risk:.4f}")
    
    # Tangency portfolio (maximum Sharpe ratio)
    tang_weights, tang_return, tang_risk = optimizer.find_tangency_portfolio(risk_free_rate=0.02/252)  # 2% annual risk-free rate
    if tang_weights is not None:
        sharpe_ratio = (tang_return - 0.02/252) / tang_risk
        print(f"📊 Tangency Portfolio: Return={tang_return:.4f}, Risk={tang_risk:.4f}, Sharpe={sharpe_ratio:.4f}")
    
    # Method 4: Risk Parity
    print("\n⚖️ Method 4: Risk Parity Optimization")
    risk_parity_optimizer = RiskParityOptimizer(mean_daily_returns, cov_matrix, tickers)
    rp_weights, rp_return, rp_risk = risk_parity_optimizer.optimize()
    if rp_weights is not None:
        print(f"⚖️ Risk Parity Portfolio: Return={rp_return:.4f}, Risk={rp_risk:.4f}")
    
    # Generate random portfolios for comparison
    print("\nGenerating random portfolios for comparison...")
    random_returns, random_risks = generate_random_portfolios(
        mean_daily_returns, cov_matrix, n_portfolios=1000
    )
    
    # Calculate individual asset metrics
    individual_returns = mean_daily_returns
    individual_risks = np.sqrt(np.diag(cov_matrix))
    
    # Create comprehensive visualization
    create_comprehensive_plot(
        # Efficient frontiers
        efficient_returns, efficient_risks,
        efficient_returns_short, efficient_risks_short,
        # Random portfolios
        random_returns, random_risks,
        # Individual assets
        individual_returns, individual_risks, tickers,
        # Special portfolios
        min_var_return, min_var_risk,
        tang_return if tang_weights is not None else None, 
        tang_risk if tang_weights is not None else None,
        rp_return if rp_weights is not None else None,
        rp_risk if rp_weights is not None else None
    )
    
    # Portfolio composition analysis
    print("\nPortfolio Composition Analysis")
    print("=" * 50)
    
    analyze_portfolio_composition(
        "Minimum Variance", min_var_weights, tickers, 
        min_var_return, min_var_risk, mean_daily_returns, cov_matrix
    )
    
    if tang_weights is not None:
        analyze_portfolio_composition(
            "Tangency (Max Sharpe)", tang_weights, tickers,
            tang_return, tang_risk, mean_daily_returns, cov_matrix
        )
    
    if rp_weights is not None:
        analyze_portfolio_composition(
            "Risk Parity", rp_weights, tickers,
            rp_return, rp_risk, mean_daily_returns, cov_matrix
        )
    
    print("\n🎉 Analysis Complete!")


def create_comprehensive_plot(eff_ret, eff_risk, eff_ret_short, eff_risk_short,
                            rand_ret, rand_risk, ind_ret, ind_risk, tickers,
                            mv_ret, mv_risk, tang_ret=None, tang_risk=None,
                            rp_ret=None, rp_risk=None):
    """Create a comprehensive visualization of all methods"""
    
    plt.figure(figsize=(14, 10))
    
    # Random portfolios (background)
    plt.scatter(rand_risk, rand_ret, alpha=0.2, s=8, c='lightgray', 
               label='Random Portfolios', zorder=1)
    
    # Efficient frontiers
    plt.plot(eff_risk, eff_ret, 'b-', linewidth=3, 
             label='Efficient Frontier (Long Only)', zorder=3)
    plt.plot(eff_risk_short, eff_ret_short, 'r--', linewidth=2, 
             label='Efficient Frontier (Short Selling)', zorder=3)
    
    # Individual assets
    colors = ['gold', 'orange', 'green', 'purple', 'brown']
    for i, ticker in enumerate(tickers):
        plt.scatter(ind_risk[i], ind_ret[i], s=150, marker='*', 
                   c=colors[i % len(colors)], edgecolors='black', linewidth=1,
                   label=f'{ticker}', zorder=4)
    
    # Special portfolios
    plt.scatter(mv_risk, mv_ret, s=200, marker='D', c='blue', 
               edgecolors='white', linewidth=2, label='Min Variance', zorder=5)
    
    if tang_ret is not None and tang_risk is not None:
        plt.scatter(tang_risk, tang_ret, s=200, marker='^', c='red',
                   edgecolors='white', linewidth=2, label='Tangency (Max Sharpe)', zorder=5)
    
    if rp_ret is not None and rp_risk is not None:
        plt.scatter(rp_risk, rp_ret, s=200, marker='s', c='green',
                   edgecolors='white', linewidth=2, label='Risk Parity', zorder=5)
    
    plt.xlabel('Risk (Standard Deviation)', fontsize=12)
    plt.ylabel('Expected Return', fontsize=12)
    plt.title(f'Comprehensive Portfolio Optimization Analysis\n({len(tickers)} Assets)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def analyze_portfolio_composition(name, weights, asset_names, portfolio_return, 
                                portfolio_risk, mean_returns, cov_matrix):
    
    print(f"\n{name} Portfolio:")
    print("-" * (len(name) + 11))
    
    metrics = calculate_portfolio_metrics(weights, mean_returns, cov_matrix)
    print(f"Expected Return: {portfolio_return:.4f}")
    print(f"Risk (Std Dev):  {portfolio_risk:.4f}")
    print(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.4f}")
    

    print("Asset Allocation:")
    for i, (asset, weight) in enumerate(zip(asset_names, weights)):
        print(f"  {asset}: {weight:.3f} ({weight*100:.1f}%)")
    
    try:
        from portfolio_optimizer import RiskParityOptimizer
        temp_optimizer = RiskParityOptimizer(mean_returns, cov_matrix, asset_names)
        risk_contrib = temp_optimizer.risk_contributions(weights)
        print("Risk Contributions:")
        for i, (asset, contrib) in enumerate(zip(asset_names, risk_contrib)):
            print(f"  {asset}: {contrib:.4f} ({contrib*100:.1f}%)")
    except:
        pass


if __name__ == "__main__":
    main()
    
