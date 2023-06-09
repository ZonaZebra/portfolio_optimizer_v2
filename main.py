import matplotlib.pyplot as plt
from config import ALPHA_VANTAGE_API_KEY
from ga_solver import ga_solver
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np

# Set API key
API_KEY = ALPHA_VANTAGE_API_KEY

# Fetch historical data for multiple stocks


def get_stock_data(symbols, start_date, end_date):
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    stock_data = []

    for symbol in symbols:
        data, _ = ts.get_daily_adjusted(symbol=symbol, outputsize='full')
        # Ensure the index is of type DatetimeIndex
        data.index = pd.to_datetime(data.index)
        data = data.sort_index(ascending=True)  # Sort the index
        data = data.loc[start_date:end_date, '5. adjusted close']
        data.name = symbol
        stock_data.append(data)

    return pd.concat(stock_data, axis=1)


def plot_pie_charts(portfolios, symbols):
    fig, axs = plt.subplots(2, 5, figsize=(15, 7))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, ax in enumerate(axs.flatten()):
        if i >= len(portfolios):
            fig.delaxes(ax)
            continue

        p = portfolios[i]

        # Check if any weights are negative
        if (p < 0).any():
            # Skip this portfolio if it has negative weights
            print(
                f'Skipping portfolio {i + 1} because it has negative weights')
            continue

        ax.pie(p, labels=symbols, autopct='%1.1f%%', startangle=90)
        # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.axis('equal')
        ax.set_title(f'Portfolio {i + 1}')

    plt.show()


def get_sp500_data(start_date, end_date):
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data, _ = ts.get_daily_adjusted(symbol='SPY', outputsize='full')
    data.index = pd.to_datetime(data.index)
    data = data.sort_index(ascending=True)
    data = data.loc[start_date:end_date, '5. adjusted close']
    data.name = 'S&P 500'
    return data


def plot_cumulative_returns(portfolios, returns, sp500_returns, symbols):
    fig, ax = plt.subplots(figsize=(12, 7))

    # Calculate the cumulative returns for each portfolio
    for i, p in enumerate(portfolios):
        cumulative_returns = (1 + (returns * p).sum(axis=1)).cumprod()
        ax.plot(cumulative_returns, label=f'Portfolio {i + 1}')

    # Calculate the cumulative returns for the S&P 500
    sp500_cumulative_returns = (1 + sp500_returns).cumprod()
    ax.plot(sp500_cumulative_returns, label='S&P 500', linestyle='--')

    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Returns')
    ax.set_title('Portfolio vs S&P 500 Performance')
    ax.legend()

    plt.show()


# Example usage
symbols = ['MSFT', 'AAPL', 'GOOGL', 'AMZN']
start_date = '2011-01-01'
end_date = '2022-12-31'

stock_data = get_stock_data(symbols, start_date, end_date)
sp500_data = get_sp500_data(start_date, end_date)

# Calculate daily returns
returns = stock_data.pct_change().dropna()
sp500_returns = sp500_data.pct_change().dropna()

# Optimize portfolios using ga_solver
portfolios = ga_solver(returns)

# Calculate expected return and risk for each portfolio
expected_returns = [np.mean(returns, axis=0).dot(p) for p in portfolios]
risks = [np.sqrt(np.dot(np.dot(p, np.cov(returns, rowvar=False)), p))
         for p in portfolios]

# Plot the efficient frontier
plt.plot(risks, expected_returns, 'o', markersize=5)
plt.xlabel('Risk (Volatility)')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.show()

plot_pie_charts(portfolios, symbols)
plot_cumulative_returns(portfolios, returns, sp500_returns, symbols)
