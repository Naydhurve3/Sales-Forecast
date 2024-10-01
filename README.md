# Sales Forecasting Project
## Project Overview
This project involves building a sales forecasting model based on historical sales data. The goal is to analyze monthly sales trends and predict future sales using ARIMA (AutoRegressive Integrated Moving Average) modeling. This model helps in understanding past sales behavior and projecting future sales to aid in better business decision-making.

## Project Steps and Rationale
### 1. Data Loading and Initial Exploration
`python`
```
data = pd.read_csv("D:/MACHINE LEARNING/CSV DATASET/train.csv")
print(data.head())
```
#### Why:

We begin by loading the dataset and printing the first few rows to understand the structure of the data. This helps us get an overview of the columns and types of information in the dataset.
### 2. Date Parsing and Handling Missing Values
`python`
```
data['Order Date'] = pd.to_datetime(data['Order Date'], format='%d/%m/%Y', errors='coerce')
if data['Order Date'].isnull().any():
    print("There are missing values in 'Order Date' after conversion:")
    print(data[data['Order Date'].isnull()])
data.dropna(subset=['Order Date'], inplace=True)
```
#### Why:

Sales data is time-sensitive, so it's crucial to parse the Order Date column into a datetime format to enable time-series analysis.
Missing or incorrectly formatted dates can cause issues during analysis. Therefore, the code identifies and removes rows where Order Date is null.
### 3. Resampling Data by Month
`python`
```
monthly_sales = data.resample('M', on='Order Date').sum()
monthly_sales.fillna(0, inplace=True)
print(monthly_sales.head())
```
#### Why:

Monthly resampling is done to aggregate sales data at the monthly level. This is a key step for identifying monthly sales trends and simplifying the time-series data for forecasting purposes.
### 4. Visualization of Monthly Sales Trends
`python`
```
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index, monthly_sales['Sales'], marker='o')
plt.title('Monthly Sales Trends')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid()
plt.show()
```
#### Why:

Visualizing the monthly sales trends helps to identify seasonal patterns, trends, and potential outliers. This is crucial for understanding the data before building a forecasting model.
### 5. Feature Engineering: Extracting Date Components
`python`
```
monthly_sales['Month'] = monthly_sales.index.month
monthly_sales['Day_of_Week'] = monthly_sales.index.dayofweek
monthly_sales['Year'] = monthly_sales.index.year
monthly_sales['Quarter'] = monthly_sales.index.quarter
print(monthly_sales.head())
```
#### Why:

Extracting additional time-based features such as the month, day of the week, year, and quarter can provide insights into the seasonality and periodic behavior of sales.
### 6. Splitting Data into Train and Test Sets
`python`
```
train_size = int(len(monthly_sales) * 0.8)
train, test = monthly_sales.iloc[:train_size], monthly_sales.iloc[train_size:]
print(f'Train Size: {len(train)}, Test Size: {len(test)}')
```
#### Why:

To evaluate the performance of the ARIMA model, we split the data into training (80%) and testing (20%) sets. This ensures the model is trained on historical data and tested on unseen data.
### 7. Building and Training the ARIMA Model
`python` 
```
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(train['Sales'], order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())
```
#### Why:

- ARIMA (1, 1, 1) is chosen to model the time series data. The three parameters (p, d, q) represent:
`p`: Number of lag observations (AutoRegressive part)
`d`: Degree of differencing (Integrated part to make the series stationary)
`q`: Size of the moving average window.
- We selected ARIMA because it is a widely used and effective model for time-series forecasting.
### 8. Model Evaluation
`python`
```
predictions = model_fit.forecast(steps=len(test))
predictions_df = pd.DataFrame(predictions, index=test.index, columns=['Predicted Sales'])
rmse = mean_squared_error(test['Sales'], predictions, squared=False)
mae = mean_absolute_error(test['Sales'], predictions)
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
```
Why:

After making predictions for the test data, the model’s performance is evaluated using RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error). These metrics give an understanding of the prediction errors and model accuracy.
### 9. Plotting Predicted vs Actual Sales
`python`
```
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Sales'], label='Training Sales', color='blue')
plt.plot(test.index, test['Sales'], label='Actual Sales', color='green')
plt.plot(predictions_df.index, predictions_df['Predicted Sales'], label='Predicted Sales', color='red')
plt.title('Sales Forecasting: Predicted vs Actual')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid()
plt.show()
```
#### Why:

This plot compares the model’s predicted sales to the actual sales on the test data. It visually illustrates how closely the predictions match reality, which is useful for validating the model's accuracy.
### 10. Forecasting Future Sales
`python`
```
future_steps = 20
future_forecast = model_fit.forecast(steps=future_steps)
future_index = pd.date_range(start=monthly_sales.index[-1] + pd.Timedelta(days=1), periods=future_steps, freq='M')
future_forecast_df = pd.DataFrame(future_forecast, index=future_index, columns=['Forecasted Sales'])
print(future_forecast_df)
```
#### Why:

The model is used to predict sales for the next 20 months. This helps businesses plan ahead by understanding potential future sales trends.
### 11. Plotting Future Sales Forecast
`python`
```
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index, monthly_sales['Sales'], label='Historical Sales', color='blue')
plt.plot(future_forecast_df.index, future_forecast_df['Forecasted Sales'], label='Future Sales Forecast', color='orange')
plt.title('Future Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid()
plt.show()
```
#### Why:

This plot displays both historical sales and future forecasts in a single graph, providing a comprehensive view of past performance and expected future sales.
## Conclusion
This project successfully demonstrates a time-series analysis and forecasting pipeline using ARIMA modeling. By carefully handling the data, visualizing trends, and making predictions, the model helps in understanding sales trends and making future forecasts. The ARIMA model was chosen due to its effectiveness in capturing time-dependent structures and trends in the sales data.
