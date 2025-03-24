import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

stock_symbol = input("Choose the stock you want to Analyse: ").upper()

stock_data = yf.download(stock_symbol, start="2024-03-21", end="2025-03-21")

stock_data['SMA_7'] = stock_data['Close'].rolling(window=7).mean()
stock_data['SMA_21'] = stock_data['Close'].rolling(window=21).mean()

first_5_rows = stock_data.head()
last_5_rows = stock_data.tail()
blank_row = pd.DataFrame([np.nan] * len(stock_data.columns)).T
blank_row.columns = stock_data.columns
combined_data = pd.concat([first_5_rows, blank_row, last_5_rows])

plt.figure(figsize=(12, 6))
plt.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue')
plt.plot(stock_data.index, stock_data['SMA_7'], label='7-Day SMA', color='red', linestyle='--')
plt.plot(stock_data.index, stock_data['SMA_21'], label='21-Day SMA', color='green', linestyle='--')

plt.title(f"Stock Price and Moving Averages for {stock_symbol}")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

print(combined_data)

stock_data.to_csv('stock_data_with_sma.csv')

print('*' * 103)

required_columns = ['Open', 'High', 'Low', 'Volume', 'SMA_7', 'SMA_21']
for col in required_columns:
    if col not in stock_data.columns:
        print(f"Error: '{col}' column not found in stock data.")
        exit()

x = stock_data[required_columns].dropna()
y = stock_data.loc[x.index, 'Close']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print('Machine Learning Results:')
print("Sample Predicted Price:\n", y_pred[:5])
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
