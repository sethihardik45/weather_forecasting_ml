# Weather Forecasting with Python

This project involves analyzing historical weather data to understand temperature trends, visualize seasonal patterns, and forecast future temperatures using machine learning models.

## Table of Contents
- [Introduction](#introduction)
- [Libraries Used](#libraries-used)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Temperature Clustering](#temperature-clustering)
- [Yearly Average Temperature](#yearly-average-temperature)
- [Seasonal Weather Analysis](#seasonal-weather-analysis)
- [Weather Forecasting with Machine Learning](#weather-forecasting-with-machine-learning)
- [Insights](#insights)
- [Conclusion](#conclusion)

## Introduction
This project analyzes historical weather data to understand temperature patterns across different seasons and years. The analysis aims to uncover insights into climate trends and forecast future temperatures using machine learning techniques.

## Libraries Used
```python
import numpy as np  # For Linear Algebra
import pandas as pd  # To Work With Data
import plotly.express as px  # For visualizations
import plotly.graph_objects as go  # For more detailed visualizations
from plotly.subplots import make_subplots  # To create subplot layouts
from datetime import datetime  # Time Series analysis
from sklearn.cluster import KMeans  # Clustering Algorithm
from sklearn.tree import DecisionTreeRegressor  # Machine Learning Model
from sklearn.model_selection import train_test_split  # Data Splitting
from sklearn.metrics import r2_score  # Model Evaluation
```

## Data Preprocessing
The dataset is loaded, and any unnecessary columns (like unnamed indices) are removed. The data is then reshaped to make it suitable for time series analysis by creating a `Date` attribute that combines the year and month.

```python
df = pd.read_csv("Weather.csv", index_col=0)
df1 = pd.melt(df, id_vars='YEAR', value_vars=df.columns[1:])
df1['Date'] = df1['variable'] + ' ' + df1['YEAR'].astype(str)
df1.loc[:, 'Date'] = df1['Date'].apply(lambda x: datetime.strptime(x, '%b %Y'))
```

## Exploratory Data Analysis (EDA)
### Temperature Through Time
A time series plot is generated to visualize temperature changes over time.

```python
fig = go.Figure(layout=go.Layout(yaxis=dict(range=[0, df1['Temprature'].max()+1])))
fig.add_trace(go.Scatter(x=df1['Date'], y=df1['Temprature']))
fig.update_layout(title='Temperature Through Timeline:', xaxis_title='Time', yaxis_title='Temperature in Degrees')
fig.show()
```

### Warmest/Coldest/Average Monthly Temperatures
A box plot is used to analyze monthly temperature variations.

```python
fig = px.box(df1, 'Month', 'Temprature')
fig.update_layout(title='Warmest, Coldest and Median Monthly Temperature')
fig.show()
```

## Temperature Clustering
Temperature data is clustered to identify different patterns.

```python
km = KMeans(3)
km.fit(df1['Temprature'].to_numpy().reshape(-1,1))
df1['Temp Labels'] = km.labels_
fig = px.scatter(df1, 'Date', 'Temprature', color='Temp Labels')
fig.update_layout(title='Temperature Clusters:', xaxis_title='Date', yaxis_title='Temperature')
fig.show()
```

## Yearly Average Temperature
The yearly average temperature is calculated and plotted to observe long-term trends.

```python
df['Yearly Mean'] = df.iloc[:, 1:].mean(axis=1)
fig = go.Figure(data=[
    go.Scatter(name='Yearly Temperatures', x=df['YEAR'], y=df['Yearly Mean'], mode='lines'),
    go.Scatter(name='Yearly Temperatures', x=df['YEAR'], y=df['Yearly Mean'], mode='markers')
])
fig.update_layout(title='Yearly Mean Temperature:', xaxis_title='Time', yaxis_title='Temperature in Degrees')
fig.show()
```

## Seasonal Weather Analysis
Temperatures are grouped into seasons and analyzed over the years.

```python
df['Winter'] = df[['DEC', 'JAN', 'FEB']].mean(axis=1)
df['Summer'] = df[['MAR', 'APR', 'MAY']].mean(axis=1)
df['Monsoon'] = df[['JUN', 'JUL', 'AUG', 'SEP']].mean(axis=1)
df['Autumn'] = df[['OCT', 'NOV']].mean(axis=1)
seasonal_df = pd.melt(df[['YEAR', 'Winter', 'Summer', 'Monsoon', 'Autumn']], id_vars='YEAR')
seasonal_df.columns = ['Year', 'Season', 'Temperature']
fig = px.scatter(seasonal_df, 'Year', 'Temperature', facet_col='Season', facet_col_wrap=2, trendline='ols')
fig.update_layout(title='Seasonal Mean Temperatures Through Years:')
fig.show()
```

## Weather Forecasting with Machine Learning
A Decision Tree Regressor is used to forecast monthly mean temperatures.

```python
dtr = DecisionTreeRegressor()
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
dtr.fit(train_x, train_y)
pred = dtr.predict(test_x)
```

### Forecasting Temperatures for 2018
The trained model is used to predict temperatures for 2018.

```python
next_Year = df1[df1['Year']==2017][['Year', 'Month']]
next_Year.Year.replace(2017, 2018, inplace=True)
temp_2018 = dtr.predict(pd.get_dummies(next_Year))
```

## Insights
- **May 1921** was the hottest month in India’s history.
- **December, January, and February** are the coldest months, typically grouped as "Winter."
- **April to August** are the hottest months, grouped as "Summer."
- Despite having four seasons, temperature clustering shows three main clusters.
- **Global Warming:** A noticeable increase in yearly mean temperatures since 1980.

## Conclusion
This project demonstrates how historical weather data can be analyzed to uncover patterns and forecast future temperatures. The findings indicate significant climate changes over the past century, particularly an increase in temperature post-1980.
