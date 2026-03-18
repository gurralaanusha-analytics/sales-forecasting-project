import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# SETUP
# -----------------------------
os.makedirs('output', exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv('medium_synthetic_retail_dataset.csv', parse_dates=['date'])

print("Data Loaded\n")
print(df.head())

# -----------------------------
# DATA CHECKS
# -----------------------------
print("\nMissing Values:\n", df.isna().sum())
print("\nDuplicates:", df.duplicated().sum())

# -----------------------------
# CLEANING
# -----------------------------
df = df.drop_duplicates()
df = df[df['sales'] >= 0]

print("\nAfter Cleaning:", df.shape)

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
df['year_month'] = df['date'].dt.to_period('M').dt.to_timestamp()

monthly_data = df.groupby('year_month', as_index=False).agg({
    'sales': 'sum',
    'units_sold': 'sum',
    'cost': 'sum'
})

monthly_data.rename(columns={'year_month': 'ds', 'sales': 'y'}, inplace=True)
monthly_data = monthly_data.sort_values('ds')

print("\nMonthly Data:\n", monthly_data.tail())

# -----------------------------
# SALES TREND
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(monthly_data['ds'], monthly_data['y'])
plt.title("Monthly Sales Trend")
plt.grid()
plt.savefig("output/sales_trend.png")
plt.close()

# -----------------------------
# DECOMPOSITION
# -----------------------------
from statsmodels.tsa.seasonal import seasonal_decompose

monthly_data.set_index('ds', inplace=True)

try:
    decomposition = seasonal_decompose(monthly_data['y'], model='additive', period=12)
    fig = decomposition.plot()
    fig.savefig("output/decomposition.png")
    plt.close()
    print("\nDecomposition completed")
except Exception as e:
    print("Decomposition failed:", e)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
train = monthly_data[:-6]
test = monthly_data[-6:]

# -----------------------------
# BASELINE MODEL
# -----------------------------
baseline_pred = np.repeat(train['y'].iloc[-1], len(test))

from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

rmse_baseline = rmse(test['y'], baseline_pred)
mape_baseline = mape(test['y'], baseline_pred)

print("\nBaseline RMSE:", rmse_baseline)
print("Baseline MAPE:", mape_baseline)

# -----------------------------
# PROPHET MODEL
# -----------------------------
from prophet import Prophet

try:
    prophet_df = train.reset_index()[['ds', 'y']]

    model = Prophet()
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=len(test), freq='MS')
    forecast = model.predict(future)

    prophet_pred = forecast.set_index('ds')['yhat'][-len(test):]

    rmse_prophet = rmse(test['y'], prophet_pred)
    mape_prophet = mape(test['y'], prophet_pred)

    print("\nProphet RMSE:", rmse_prophet)
    print("Prophet MAPE:", mape_prophet)

except Exception as e:
    print("Prophet failed:", e)
    prophet_pred = baseline_pred

# -----------------------------
# SAVE RESULTS
# -----------------------------
results = pd.DataFrame({
    'Model': ['Baseline', 'Prophet'],
    'RMSE': [rmse_baseline, rmse(test['y'], prophet_pred)],
    'MAPE': [mape_baseline, mape(test['y'], prophet_pred)]
})

results.to_csv("output/model_results.csv", index=False)

# -----------------------------
# FINAL FORECAST PLOT
# -----------------------------
plt.figure(figsize=(10,5))

plt.plot(train.index, train['y'], label='Train')
plt.plot(test.index, test['y'], label='Actual')

plt.plot(test.index, baseline_pred, label='Baseline')
plt.plot(test.index, prophet_pred, label='Prophet')

plt.legend()
plt.title("Forecast Comparison")

plt.savefig("output/final_forecast.png")
plt.close()

print("\nAll outputs saved in 'output/' folder ✅")

# -----------------------------
# KPI CALCULATIONS
# -----------------------------

total_revenue = df['sales'].sum()
total_units = df['units_sold'].sum()
avg_order_value = total_revenue / total_units

monthly_growth = monthly_data['y'].pct_change().mean() * 100

print("\n--- KPI SUMMARY ---")
print("Total Revenue:", total_revenue)
print("Total Units Sold:", total_units)
print("Average Order Value:", avg_order_value)
print("Average Monthly Growth (%):", monthly_growth)
import plotly.express as px

import plotly.express as px

# Forecast plot
fig = px.line(forecast, x='ds', y='yhat', title='Sales Forecast')

# Save interactive HTML
fig.write_html('output/forecast_plot.html')

# Optional: save static PNG
fig.write_image('output/forecast_plot.png')

# Save KPIs to CSV
import pandas as pd
kpi_data = {
    'Total Revenue': [total_revenue],
    'Total Units Sold': [total_units],
    'Average Order Value': [avg_order_value],
    'Average Monthly Growth (%)': [monthly_growth]  # <-- corrected
}
pd.DataFrame(kpi_data).to_csv('output/KPI_summary.csv', index=False)