---
layout: post
title:      "Project 4: Time Series"
date:       2021-04-08 00:52:47 +0000
permalink:  project_4_time_series
---

## Introduction

For this project, I am acting as a consultant for a fictional real-estate investment firm, and I will be forecasting real estate prices of various zip codes using data from Zillow. The goal of the project is to analyze, model, and predict returns in order to recommend the best 5 zipcodes to invest in Gwinnett County, Georgia where I currently reside. In this blog post, I will share the basic knowledge about time series and how I predicted the house price using ARIMA/SARIMAX models

The dataset I used was provided by Flatiron School, which was obtained from Zillow Research Page (https://www.zillow.com/research/data/). It contains nearly 15,000 lines with basic information such as city, state, county, size rank from April 1996 - April 2018. The Gwinnett County portion has 4240 line with 16 zip codes.

## Data preprocessing

Firstly, I select the city and reshape the data frame from wide to long format by creating a function ...
```
def melt_data(df):
    
    '''Takes a dataframe with datetime data that is in wide format and melts it into long format'''
    
    melted = pd.melt(df, id_vars=['Zipcode'], var_name='time')
    melted['time'] = pd.to_datetime(melted['time'], infer_datetime_format=True)
    melted = melted.dropna(subset=['value'])
    return melted
```
...and then transform to time series data
```
gwinnett_melted = melt_data(gwinnett_copy)
gwinnett_melted = gwinnett_melted.set_index('time')
```

Next, I want to observe the price ranges for each zipcode.
```
fig, ax = plt.subplots(figsize = (12,12))
sns.boxplot(x = 'Zipcode', y = 'value', data = gwinnett_melted, ax = ax)
plt.xticks(fontsize = 11)
plt.xlabel('Zip Code', fontsize = 11)
plt.ylabel('Value', fontsize = 11)
plt.yticks(fontsize = 11)
```

And this is the plot resulted from above lines
![](https://raw.githubusercontent.com/helenpham0229/Flatiron-Project-4-Housing-Price-Time-Series/main/images%20from%20notebook/gwinnett%20zip%20codes.png)

Then, I want to group my data by zipcode, which makes it easier to select each zipcode later for further analysis and modeling process
```
gwinnett_grouped = gwinnett_melted.groupby(['Zipcode'], as_index = True).resample('M').sum()
```

## Time Series Analysis and Modeling
I created 4 functions to help me observe the characteristics of the original dataset and build models for each zipcode.

**Function 1: Stationarity check with Dickey-Fuller test**

Most time series models work on the assumption that the time series are stationary which means its statistical properties such as mean, variance, etc. remain constant over time. Ideally, we want to have a stationary time series for modeling. Dickey-Fuller test can be used to test if a time series is stationary or not. For each zip code times series that I have for Gwinnett County, I run this function and none of them is stationary. However, when I run this function with the residuals of each time series, they are all stationary.
```
def stationarity_check(ts):
    '''Takes time series dataframe and returns with results for Dickey-Fuller test'''
    dftest = adfuller(ts.dropna())
    print('\nResults of Dickey-Fuller Test: \n')
    dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
```

**Function 2: Time Series Decomposition**

Time series decomposition is a mathematical procedure that transforms a time series into multiple different time series. The original time series is often split into three component series:
- Seasonal: Patterns that repeat within a fixed period. 
- Trend: The underlying trend of the metrics. 
- Random: Also called "noise", "irregular", or "remainder", this is the residual of the original time series after the seasonal and trend series are removed.
```
def decomposition(ts):
    decomposition = seasonal_decompose(ts)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    plt.figure(figsize=(10,7))
    plt.subplot(511)
    plt.plot(ts, label='Original', color='blue')
    plt.legend(loc='best')
    plt.subplot(512)
    plt.plot(trend, label='Trend', color='blue')
    plt.legend(loc='best')
    plt.subplot(513)
    plt.plot(seasonal,label='Seasonality', color='blue')
    plt.legend(loc='best')
    plt.subplot(514)
    plt.plot(residual, label='Residuals', color='blue')
    plt.legend(loc='best')
    plt.tight_layout()
```

Now, one of your decomposition plots should look similar to the one below.

![](https://raw.githubusercontent.com/helenpham0229/Flatiron-Project-4-Housing-Price-Time-Series/main/images%20from%20notebook/decomposition%20example.png)

In all decomposition plots for each zipcode, I notice that there is a strong seasionality component, a downward trend from 2008-2012, and an upward trend since mid - 2012.

**Function 3: Run auto arima using SARIMAX**

Since our dataset is not stationary, and there is a seasonal component, it would be reasonable to use SARIMA model - Seasonal ARIMA (Seasonal Autoregressive Integrated Moving Averages with exogenous regressors)

Per the formula SARIMA(p,d,q)x(P,D,Q,s), the parameters for these types of models are as follows:
- p and seasonal P: indicate the number of autoregressive terms (lags of the stationarized series)
- d and seasonal D: indicate differencing that must be done to stationarize series
- q and seasonal Q: indicate number of moving average terms (lags of the forecast errors)
- s: indicates periodicity of the time series (4 for quarterly, 12 for yearly)

In this model, I use AIC to select best set of parameters. AIC is a single number score that can be used to determine which of multiple models is most likely to be the best model for a given dataset. It estimates models relatively, meaning that AIC scores are only useful in comparison with other AIC scores for the same dataset. A lower AIC score is better.

```
def best_parameters(ts):
    best_orders = pm.auto_arima(ts, start_p = 0, start_q = 0, max_p = 4, max_q = 4, m = 12, seasonal = True, 
                                   stationary = False, stepwise = True, trend = 'ct', suppress_warnings = True, 
                                   trace = False, error_action = 'ignore')
    best_model = SARIMAX(ts, order = best_orders.order, seasonal_order = best_orders.seasonal_order).fit()
    best_parameters = []
    best_parameters.append([best_orders.order, best_orders.seasonal_order, best_model.aic]) 
    print('ARIMA {} x {}, AIC Calculated: {}'.format(best_orders.order, best_orders.seasonal_order, best_model.aic))
    print(best_model.summary())
    return best_model
```

Your results should look similar to this...
![](https://raw.githubusercontent.com/helenpham0229/Flatiron-Project-4-Housing-Price-Time-Series/main/images%20from%20notebook/SARIMAX%20example.PNG)

**Function 4: Forecasting**

This function helps us to get one-step-ahead forecast for model, calculates root of mean squared error, makes future predictions , plots results and provides forecasted value with 95% confidence interval

```
# get predictions for the best model
def get_predictions(ts, model, steps=60, plot=True, show=True):
    '''Enter time series and model
    Gets one-step-ahead forecast for model
    Calculates Root of Mean Squared Error
    Makes future predictions for number of steps passed as parameter (default is 60 or 5 years), 
    Plots results, 
    Provides forecasted value with confidence interval'''
    
    # Get preditions from model for the last 20% of data period
    pred = model.get_prediction(start='2013-11-30', dynamic=False)
    conf = pred.conf_int()

    # Plot observed and predicted values with confidence interval
    if plot:
        ax = ts['1996':].plot(label='Observed', figsize=(8, 8))
        pred.predicted_mean.plot(ax=ax, label='One-step-ahead Forecast', alpha=.5)
        ax.fill_between(conf.index,
                        conf.iloc[:, 0],
                        conf.iloc[:, 1], color='g', alpha=.3,
                        label='Confidence Interval')
        ax.set_ylabel('Value')
        ax.set_xlabel('Year')
        plt.title('Observations vs Predictions')
        ax.legend()
        plt.show()
        
    # Compare real and predicted values to validade model and compute the rmse
    predicted = pred.predicted_mean
    real = ts['2017-01-01':].value
    mse = np.square(np.subtract(real,predicted)).mean()
    rmse = math.sqrt(mse)
    
    if show:
        print(f'The RMSE of our forecast is {round(rmse, 2)}.' + '\n')
        
    # Get forecast and confidence interval for steps ahead in future
    future = model.get_forecast(steps=steps, dynamic=True)
    future_conf = future.conf_int()

    # Plot future forecast with confidence interval
    if plot:
        ax = ts['1996':].plot(label='Observed', figsize=(8, 8))
        future.predicted_mean.plot(ax=ax, label='Forecast')
        ax.fill_between(future_conf.index,
                        future_conf.iloc[:, 0],
                        future_conf.iloc[:, 1], color='g', alpha=.3)
        ax.fill_betweenx(ax.get_ylim(), 
                         pd.to_datetime('2023-04-30'), 
                         predicted.index[-1], alpha=.1, zorder=-1)
        ax.set_xlabel('Year')
        ax.set_ylabel('Value')
        plt.title('Future Forecast')
        ax.legend()
        plt.show()
        
    # show prediction for end of step-period (in this case in 5 years future time)
    forecast = future.predicted_mean[-1]
    upper = future_conf.iloc[-1,1]
    lower = future_conf.iloc[-1,0]
    predictions = {}
    predictions['forecast'] = forecast
    predictions['upper'] = upper
    predictions['lower'] = lower
    predictions = pd.DataFrame.from_dict(predictions, orient='index', columns=['Return at End of Forecast'])
  
    # calculate return percentages
    price_2018 = ts.loc['2018-04-30']
    forecast_2023 = forecast
    forecast_lower = lower
    forecast_upper = upper
    return_percentage_predictions = {}
    predicted_percent_change = ((forecast_2023- price_2018) / price_2018)*100
    best_percent_change = ((forecast_upper - price_2018) / price_2018)*100
    worst_percent_change = ((forecast_lower - price_2018) / price_2018)*100
    return_percentage_predictions['predicted_percent_change'] = predicted_percent_change
    return_percentage_predictions['best_case'] = best_percent_change
    return_percentage_predictions['worst_case'] = worst_percent_change
    return_percentage_predictions = pd.DataFrame.from_dict(return_percentage_predictions,orient='index')
    
    if show:
        print(predictions)
        
        print(return_percentage_predictions)
```
After running this function, the results for each zipcode should look like this image below. Note that I use RMSE, a measure of the differences between values predicted by the model and the values observed to see how accurate my models are.

![](https://raw.githubusercontent.com/helenpham0229/Flatiron-Project-4-Housing-Price-Time-Series/main/images%20from%20notebook/Validation%20plot.PNG)

![](https://raw.githubusercontent.com/helenpham0229/Flatiron-Project-4-Housing-Price-Time-Series/main/images%20from%20notebook/Forecast%20Example.PNG)

## Conclusion
Here are the top 5 zipcodes based on ROI

![alt text](https://raw.githubusercontent.com/helenpham0229/Flatiron-Project-4-Housing-Price-Time-Series/main/images%20from%20notebook/summary%20table.PNG)

In conclusion, this is a very basic time series model. In the near future,  I would like to explore more relevant real-esate time series topics such as:
- How did COVID-19 affect the housing market in terms of price, inventory, supply, and demand?
- How private and public school district affect the sale prices in different areas?


