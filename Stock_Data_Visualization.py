from curses.ascii import isdigit
from matplotlib.lines import lineStyles
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from datetime import timedelta


well_known_stocks = {
    "Apple Inc.": "AAPL",
    "Microsoft Corporation": "MSFT",
    "Amazon.com, Inc.": "AMZN",
    "Alphabet Inc. (Class A)": "GOOGL",
    "Alphabet Inc. (Class C)": "GOOG",
    "Tesla, Inc.": "TSLA",
    "Berkshire Hathaway Inc. (Class B)": "BRK.B",
    "NVIDIA Corporation": "NVDA",
    "Meta Platforms, Inc.": "META",
    "Johnson & Johnson": "JNJ",
    "Visa Inc.": "V",
    "JPMorgan Chase & Co.": "JPM",
    "Procter & Gamble Co.": "PG",
    "UnitedHealth Group Incorporated": "UNH",
    "Mastercard Incorporated": "MA",
    "Walmart Inc.": "WMT",
    "The Home Depot, Inc.": "HD",
    "Chevron Corporation": "CVX",
    "Pfizer Inc.": "PFE",
    "Coca-Cola Company": "KO",
    "PepsiCo, Inc.": "PEP",
    "Intel Corporation": "INTC",
    "The Walt Disney Company": "DIS",
    "Exxon Mobil Corporation": "XOM",
    "AbbVie Inc.": "ABBV",
    "Bank of America Corporation": "BAC",
    "Adobe Inc.": "ADBE",
    "Netflix, Inc.": "NFLX",
    "Cisco Systems, Inc.": "CSCO",
    "AT&T Inc.": "T",
    "Verizon Communications Inc.": "VZ",
    "Merck & Co., Inc.": "MRK",
    "PayPal Holdings, Inc.": "PYPL",
    "Salesforce, Inc.": "CRM",
    "McDonald's Corporation": "MCD",
    "Nike, Inc.": "NKE",
    "Comcast Corporation": "CMCSA",
    "Amgen Inc.": "AMGN",
    "Boeing Company": "BA",
    "International Business Machines Corporation": "IBM",
    "Thermo Fisher Scientific Inc.": "TMO",
    "Texas Instruments Incorporated": "TXN",
    "Citigroup Inc.": "C",
    "Goldman Sachs Group, Inc.": "GS",
    "Starbucks Corporation": "SBUX",
    "General Electric Company": "GE",
    "Eli Lilly and Company": "LLY",
    "3M Company": "MMM",
    "Bristol-Myers Squibb Company": "BMY",
    "American Express Company": "AXP"
}

def print_ticket_symbols():
    sorted_by_keys = dict(sorted(well_known_stocks.items(), reverse=False))
    for company, ticker in sorted_by_keys.items():
       print(f"{company}: {ticker}")


# Function to calculate and display stock statistics
def calculate_stock_statistics(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")

        if data.empty:
            print(f"No data found for ticker symbol '{ticker}'. Please check the symbol and try again.")
            return

        df = pd.DataFrame(data)

        # Calculate minimum, maximum, and average closing prices
        min_close = df['Close'].min()
        max_close = df['Close'].max()
        avg_close = df['Close'].mean()

        # Print the calculated values
        print(f"Minimum Closing Price for {ticker}: ${min_close:.2f}")
        print(f"Maximum Closing Price for {ticker}: ${max_close:.2f}")
        print(f"Average Closing Price for {ticker}: ${avg_close:.2f}")

    except Exception as e:
        print(f"An error occurred: {e}")


def perform_random_forest_prediction(ticker):
    try:
        # Retrieve historical market data
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")

        # Check if data is retrieved
        if data.empty:
            print(f"No data found for ticker symbol '{ticker}'. Please check the symbol and try again.")
            return

        # Prepare data for Random Forest
        df = pd.DataFrame(data)
        df['Date'] = np.arange(len(df.index))  # Creating a sequence of numbers for dates
        X = df['Date'].values.reshape(-1, 1)  # Independent variable (Date as a sequence)
        y = df['Close'].values.reshape(-1, 1)  # Dependent variable (Closing prices)
     
        # Train the Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y.ravel())  # Flatten y to avoid warnings
        y_pred = model.predict(X)

        # Future predictions for the next 90 days
        future_days = 90
        future_dates = np.arange(len(df.index), len(df.index) + future_days).reshape(-1, 1)
        future_predictions = model.predict(future_dates)

        # Create a new index for future dates
        last_date = df.index[-1]
        future_index = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]


        # Plotting everything on one graph
        plt.figure(figsize=(12, 8))

        # Plot actual closing prices
        plt.plot(df.index, df['Close'], label=f"{ticker} Close Price", color='green', linestyle='dotted')

        # Plot Random Forest regression line on historical data
        # plt.plot(df.index, y_pred, label='Random Forest Regression', color='red', linestyle='dotted')

        # Plot future predicted prices
        plt.plot(future_index, future_predictions, label='Future Predicted Prices', color='blue', linestyle='--')

        # Graph details
        plt.title(f"Random Forest Prediction on {ticker} Stock Price")
        plt.xlabel("Date")
        plt.ylabel("Closing Price (USD)")
        plt.legend()
        plt.grid(True)

        # Display the plot
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


# Function to perform linear regression on stock prices
def perform_linear_regression(ticker):
    try:
        # Retrieve historical market data
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")

        # Check if data is retrieved
        if data.empty:
            print(f"No data found for ticker symbol '{ticker}'. Please check the symbol and try again.")
            return

        # Prepare data for linear regression
        df = pd.DataFrame(data)

        # print(data)

        df['Date'] = np.arange(len(df.index))  # Creating a sequence of numbers for dates
        X = df['Date'].values.reshape(-1, 1)  # Independent variable (Date as a sequence)
        # print(X)
        y = df['Close'].values.reshape(-1, 1)  # Dependent variable (Closing prices)
        # print(y)
     
        # Perform linear regression
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Future predictions for the next 90 days
        future_days = 90
        future_dates = np.arange(len(df.index), len(df.index) + future_days).reshape(-1, 1) # future dates 0-365 ,giving us a single column instead of 2-D array
        # print(future_dates)
        future_predictions = model.predict(future_dates) # linear regression on future dates

        # Create a new index for future dates
        last_date = df.index[-1]
        future_index = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]


        # Create a figure with two subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))

        # First subplot: Actual closing prices and linear regression line
        axs[0].plot(df.index, df['Close'], label=f"{ticker} Close Price", color='green', linestyle='dotted')
        axs[0].plot(df.index, y_pred, label='Linear Regression', color='red', linestyle='dotted')
        axs[0].set_title(f"Linear Regression on {ticker} Stock Price Over the Last Year")
        axs[0].set_xlabel("Date")
        axs[0].set_ylabel("Closing Price (USD)")
        axs[0].legend()
        axs[0].grid(True)

        # Second subplot: Future predicted prices
        axs[1].plot(future_index, future_predictions, label='Future Predicted Prices', color='blue', linestyle='--')
        axs[1].set_title(f"Linear Regression on {ticker} Stock Price for the Next {future_days} Days")
        axs[1].set_xlabel("Date")
        axs[1].set_ylabel("Closing Price (USD)")
        axs[1].legend()
        axs[1].grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Display the plots
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")



# Function to retrieve and plot stock data
def plot_stock_data(ticker):
    try:
        # Retrieve historical market data
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")

        # Check if data is retrieved
        if data.empty:
            print(f"No data found for ticker symbol '{ticker}'. Please check the symbol and try again.")
            return

        # Convert to DataFrame if not already
        df = pd.DataFrame(data)

        # Plot the closing price
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['Close'], label=f"{ticker} Close Price", color='blue', linestyle = '--')
        plt.title(f"{ticker} Stock Price Over the Last Year")
        plt.xlabel("Date")
        plt.ylabel("Closing Price (USD)")
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


# Main loop
while True:
    print("----Welcome to Stock Data Visualization----")
    print("1. Plot stock data")
    print("2. Print well-known stock ticker symbols")
    print("3. Calculate stock statistics (min, max, avg)")
    print("4. Perform linear regression on stock prices")
    print("5. Perform random forest regression on stock prices")
    print("6. Exit")
    option = input("Enter your option: ")

    if option == "1":
        ticker = input("Enter the stock ticker symbol (e.g., AAPL for Apple): ").upper()
        plot_stock_data(ticker)
    elif option == "2":
        print_ticket_symbols()
    elif option == "3":
        ticker = input("Enter the stock ticker symbol (e.g., AAPL for Apple): ").upper()
        calculate_stock_statistics(ticker)
    elif option == "4":
        ticker = input("Enter the stock ticker symbol (e.g., AAPL for Apple): ").upper()
        perform_linear_regression(ticker)
    elif option == "5":
        ticker = input("Enter the stock ticker symbol (e.g., AAPL for Apple): ").upper()
        perform_random_forest_prediction(ticker)
    elif option == "6":
        print("Exiting the program. Goodbye!")
        break
    else:
        print("Invalid option. Please enter 1, 2, 3, 4, 5 or 6.")




