import yfinance as yf
import numpy as np

tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
    "SPY", "QQQ", "VTI", "ARKK", "DIA", "XOM", "CVX", "BA", "CAT", "GE",
    "JNJ", "PFE", "UNH", "TSM", "NFLX", "INTC", "AMD", "CRM", "NKE", "WMT"
]

def get_daily_returns(ticker, start="2024-01-01", end="2025-01-01"):
    data = yf.download(ticker, start=start, end=end, progress=False)

    if 'Close' not in data.columns or len(data) < 2:
        return None, None  # Return None if data is insufficient

    close_prices = data['Close'].values
    dates = data.index[1:].tolist()  # Convert to list of timestamps

    returns = [(close_prices[i+1]-close_prices[i])/close_prices[i] for i in range(len(close_prices)-1) ]
    returns = [returns[i][0] for i in range (len(returns))]  # Convert to a list of floats
    # print(returns)
    # print("okokokokookokok------")
    return dates, returns

# import numpy as np

def get_daily_returns_for_multiple_tickers(tickers, start="2024-01-01", end="2025-01-01"):
    returns_list = []

    for ticker in tickers:
        try:
            dates, returns = get_daily_returns(ticker, start, end)
            if returns is None or len(returns) < 1:
                print(f"Skipping {ticker}: insufficient data.")
                continue

            returns_list.append(returns[0])

        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    if not returns_list:
        raise ValueError("No valid return data found for any ticker.")

    return returns_list


# Example usage
if __name__ == "__main__":
    daily_returns_data = get_daily_returns_for_multiple_tickers(tickers)
    print(daily_returns_data[0].shape)  # Display shape of the returns array
    #print(daily_returns_data)
    # # Display example: print first 5 returns for each ticker
    # for ticker, data in daily_returns_data.items():
    #     print(f"{ticker} - First 5 daily returns:")
    #     for date, ret in zip(data["dates"][:5], data["returns"][:5]):
    #         print(f"  {date.date()} : {ret:.4f}")
    #     print()
