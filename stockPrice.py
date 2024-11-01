import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error


st.write(
    "Welcome to my project portfolio.\n\n"
    "In this application, I utilize machine learning techniques to provide insights into stock price movements.\n\n"
    "#### Key Features :\n"
    "- **LSTM Model** : A Long Short-Term Memory (LSTM) model is employed to predict future stock prices based on historical data.\n"
    "- **Interactive Interface** : Built with **Streamlit**\n"
    "- **Data Handling** : I use **pandas** and **numpy** for efficient data manipulation and analysis.\n"
    "- **Visualization** : Interactive visualizations with **Plotly** enable exploration of stock trends and forecasts.\n"
    "- **Performance Metrics**: Key metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) are displayed to assess prediction accuracy.\n\n"
    "#### Libraries Used :\n"
    "- **yfinance**: This library retrieves real-time stock data from Yahoo Finance.\n"
    "- **TensorFlow/Keras** : These frameworks are used for designing and training the LSTM model, incorporating **Dense** and **Dropout** layers.\n\n"
    "I invite you to explore the application and gain insights into stock price forecasting.\n\n"
    "Noë Dréau - 2024"
)

# Application title and description
st.title("Stock Price Forecasting and Comparison")
st.write("""
 
You can choose the stocks you're interested in and set the forecasting period for a personalized analysis.
""")


# List of popular tickers
default_tickers = {
    'Google': 'GOOGL',
    'Apple': 'AAPL',
    'Amazon': 'AMZN',
    'Microsoft': 'MSFT',
    'Tesla': 'TSLA',
    'Meta (Facebook)': 'META',
}

# Text input for custom ticker
custom_ticker = st.text_input("Enter a custom ticker of any additional company")

# Multiselect for choosing among default popular companies
selected_tickers = st.multiselect(
    "Select companies to compare", 
    list(default_tickers.keys()), 
    default=['Google', 'Apple']
)

# Convert selected company names to ticker symbols
tickers = [default_tickers[company] for company in selected_tickers]

# Add the custom ticker if provided by the user
if custom_ticker:
    tickers.append(custom_ticker)

# Sidebar options for analysis period
st.sidebar.header("Analysis Options")
period_option = st.sidebar.selectbox(
    "Select historical analysis period",
    ['6mo', '1y', '5y', 'max']
)

# User-defined forecast days
forecast_days = st.sidebar.number_input("Number of days to forecast", min_value=3, max_value=365, value=14)

# User-defined moving average window
moving_average_window = st.sidebar.number_input("Moving Average Window (days)", min_value=1, max_value=100, value=7)

# Function to fetch and clean stock data
def fetch_stock_data(ticker, period):
    """Fetch historical stock data for a given ticker and period."""
    try:
        ticker_data = yf.Ticker(ticker)
        df = ticker_data.history(period=period)
        df = df[['Close']].dropna()
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# Function to prepare data for the LSTM model
def prepare_data(data, time_step=1):
    """Prepare data for LSTM by creating sequences."""
    X, y = [], []
    for i in range(len(data) - time_step):
        a = data[i:(i + time_step), 0]  # Previous time steps
        X.append(a)
        y.append(data[i + time_step, 0])  # Next time step
    return np.array(X), np.array(y)

# Function to train an LSTM model and forecast into the future
def forecast_lstm(data, forecast_days):
    """Train LSTM model and forecast future stock prices."""
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    # Prepare data for LSTM
    time_step = 30  # Number of previous days to consider
    X, y = prepare_data(scaled_data, time_step)

    # Reshape X to be [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Fit the model
    history = model.fit(X, y, epochs=100, batch_size=32, verbose=0)

    # Forecasting
    forecast_list = []
    last_data = scaled_data[-time_step:]  # Last data points for forecasting
    for _ in range(forecast_days):
        last_data = last_data.reshape((1, time_step, 1))  # Reshape for LSTM
        predicted = model.predict(last_data)
        forecast_list.append(predicted[0][0])
        # Update last_data for the next prediction
        last_data = np.append(last_data[:, 1:, :], predicted.reshape(1, 1, 1), axis=1)  # Append predicted value

    # Inverse transform to get actual values
    forecast_values = scaler.inverse_transform(np.array(forecast_list).reshape(-1, 1))

    return forecast_values.flatten(), history.history['loss']  # Return as a flat array and training loss

# Retrieve and display data
if tickers:
    st.header("Closing Price Comparison")

    # Initialize an empty DataFrame to store stock data
    data = pd.DataFrame()

    # Loop through each ticker to fetch data and append closing prices to DataFrame
    for ticker in tickers:
        ticker_data = fetch_stock_data(ticker, period_option)
        if not ticker_data.empty:
            data[ticker] = ticker_data['Close']
    
    # Fill any missing dates
    data = data.fillna(method="ffill").fillna(method="bfill")

    # Calculate moving average
    moving_average = data.rolling(window=moving_average_window).mean()

    # Display the chart with closing prices and moving average for all selected companies
    fig = go.Figure()
    for ticker in tickers:
        fig.add_trace(go.Scatter(x=data.index, y=data[ticker], mode='lines', name=f'{ticker} Historical Data', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=moving_average.index, y=moving_average[ticker], mode='lines', name=f'{ticker} {moving_average_window}-Day MA', line=dict(color='green', dash='dot')))
    
    fig.update_layout(title="Stock Prices with Moving Average", xaxis_title="Date", yaxis_title="Price", legend=dict(x=0, y=1))
    st.write("Nota : "
    "A moving average is a method used to smooth out price data. It calculates the average price over a specific period.\n\n"
    "A greater moving average is less sensitive to recent price changes, providing a clearer view of the overall trend."
    )
    st.plotly_chart(fig)

    # Forecast for each ticker
    st.subheader("Forecasted Future Prices")
    for ticker in tickers:
        st.write(f"**Future Forecast for {ticker}**")
        if len(data[ticker].dropna()) < 30:  # Ensure there's enough data for LSTM
            st.write(f"Insufficient data for {ticker} to perform accurate forecasting.")
            continue

        # Forecasting into the future beyond last date in dataset
        predicted_prices, training_loss = forecast_lstm(data[ticker].dropna(), forecast_days)

        # Calculate forecast dates
        last_date = data.index[-1]  # Get the last date from the historical data
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

        # Display forecasted prices as a list with actual dates
        for i, price in enumerate(predicted_prices):
            st.write(f"{forecast_dates[i].date()}: ${price:.2f}")

        # Combine historical and forecast data for plotting
        combined_data = data[[ticker]].copy()
        forecast_df = pd.DataFrame(predicted_prices, index=forecast_dates, columns=[ticker])
        combined_data = pd.concat([combined_data, forecast_df])

        # Plot combined historical and forecast data with different colors
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data[ticker], mode='lines', name='Historical Data', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df[ticker], mode='lines', name='Forecast', line=dict(color='orange')))
        fig.update_layout(title=f"{ticker} Price Forecast", xaxis_title="Date", yaxis_title="Price", legend=dict(x=0, y=1))
        st.plotly_chart(fig)

        # Display training loss
        st.subheader("Training Loss")
        st.line_chart(training_loss)

        # Calculate and display error metrics
        actual_prices = data[ticker].tail(forecast_days).values
        mae = mean_absolute_error(actual_prices, predicted_prices)
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        st.write(f"**Mean Absolute Error (MAE)**: ${mae:.2f}")
        st.write(f"**Root Mean Squared Error (RMSE)**: ${rmse:.2f}")

    # Additional company information
    st.sidebar.subheader("Company Information")
    for ticker in tickers:
        info = yf.Ticker(ticker).info
        st.sidebar.write(f"**{info.get('longName', ticker)}**")
        st.sidebar.write(f"Sector: {info.get('sector', 'N/A')}")
        st.sidebar.write(f"Industry: {info.get('industry', 'N/A')}")
        st.sidebar.write(f"Website: [Visit site]({info.get('website', '#')})")
        st.sidebar.write("---")
else:
    st.write("Please select or enter at least one ticker to display data.")


