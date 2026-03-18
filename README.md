## Sales Forecasting Mini Project ##
This project focuses on building a time series forecasting pipeline to predict retail sales using historical data and provide actionable business insights.

## Objective ##
Analyze sales trends, identify seasonality, and forecast future sales using statistical and machine learning techniques.

## Data Preparation ##
Removed duplicate records
Handled missing values
Filtered negative sales values
Aggregated daily data into monthly format

## Analysis ##
Visualized monthly sales trends
Performed time series decomposition

## Models Used ##
Baseline Model 
Prophet Model

## Evaluation Metrics ##
RMSE (Root Mean Squared Error)
MAPE (Mean Absolute Percentage Error)

## Evaluation Metrics ##
| Model    | RMSE   | MAPE (%) |
| -------- | ------ | -------- |
| Baseline | 71,452 | 25.68    |
| Prophet  | 38,281 | 15.82    |

## Key Performance Indicators (KPIs) ##
Total Revenue: $7,607,473.52
Total Units Sold: 325,411
Average Order Value: $23.38
Average Monthly Growth: 1.92%

## Outputs ##
All outputs are saved in the output/ folder:
forecast_plot.html – Interactive forecast plot
forecast_plot.png – Static forecast plot
KPI_summary.csv – KPI summary table

## Tools & Technologies ##
Python, Pandas, NumPy, Matplotlib, Seaborn, Prophet, Plotly

## Business Insights ##
Sales show a strong upward trend during Nov–Dec, indicating seasonal demand
Promotional activities (promo_flag) positively impact sales volume
Certain product categories contribute disproportionately to total revenue, highlighting key business drivers
Monthly sales fluctuations indicate opportunities for inventory and demand planning
Regional variation in sales suggests scope for targeted marketing strategies
