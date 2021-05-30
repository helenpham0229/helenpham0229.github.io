---
layout: post
title:      "NYC Crimes - Time Series Forecasting"
date:       2021-05-30 00:32:46 +0000
permalink:  nyc_crimes_-_time_series_forecasting
---

## Introduction

For my capstone project, I decided to analyze and build a time series model predicting the number of crimes in the next 5 years, separated by borough. I obtained the dataset from [NYPD website](https://catalog.data.gov/dataset/nypd-arrests-data-historic) with all types of information about crime incidents from 2006 through 2019. After cleaning crime descriptions, locations, data types, and null values, I ended up with approximately 5 millions incidents from beginning of 2006 through the end of 2019. Then the data was grouped by month and by borough. Each borough set was split into training set from 2006 through 2016, and testing set from 2017 through 2019.

In order to create my best-fit model, I chose to go with root mean squared error (RMSE) to show how inaccurate my predictions would be. This metric was captured with each model, with final results at the end. The purpose of this post is to show a quick run-through of Holt-Winters Exponential Smoothing, SARIMAX and Facebook Prophet. You can check out my [repo here](https://github.com/helenpham0229/Flatiron-Capstone-NYC-Crimes) for more details on how I scrubbed and explored data, decomposed time series, and checked stationarity component.

## SARIMAX
Past time points of time series data can impact current and future time points. ARIMA models take this concept into account when forecasting current and future values. ARIMA uses a number of lagged observations of time series to forecast observations. A weight is applied to each of the past term and the weights can vary based on how recent they are. SARIMA(X) is an extension of ARIMA that explicitly supports univariate time series data with a seasonal component. It captures moving averages throughout time series, taking into account the data’s trend, seasonality, and noise. 

Below is a function to do a param search for SARIMAX order and seasonal_order.

```
def best_parameters(ts):
    
    '''Enter time series. This function will let auto_arima search for hyperparameters, then SARIMAX will fit these 
    hyperparameters to get the best model with lowest AIC'''
    
    best_orders = pm.auto_arima(ts, start_p = 0, start_q = 0, max_p = 2, max_q = 2, m = 12, seasonal = True, 
                                   stationary = False, stepwise = True, trend = 'ct', suppress_warnings = True, 
                                   trace = False, error_action = 'ignore')
    best_model = SARIMAX(ts, order = best_orders.order, seasonal_order = best_orders.seasonal_order).fit()
    best_parameters = []
    best_parameters.append([best_orders.order, best_orders.seasonal_order, best_model.aic]) 
    print('ARIMA {} x {}, AIC Calculated: {}'.format(best_orders.order, best_orders.seasonal_order, best_model.aic))
    print(best_model.summary())
    return best_model
```

The next function gets one-step-ahead forecast and compares it to the actual data points. In addition, it also makes future predictions for the number of steps passed as parameters, plot results with 95% confidence interval, and calculates RMSE & percentage change.

```
def get_predictions(ts, model, steps = 84, plot=True, show=True):
    
    '''Parameters: time series dataframe, model, steps, plot, and show.'''
    
    # Get preditions from model for the last 3 years of data period
    pred = model.get_prediction(start='2017-01-31', dynamic=False)
    conf = pred.conf_int()

    # Plot observed and predicted values with confidence interval
    if plot:
        ax = ts['2006':].plot(label='Observed', figsize=(10, 8))
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
    real = ts['2017-01-31':].Crime_Number
    mse = np.square(np.subtract(real,predicted)).mean()
    rmse = math.sqrt(mse)
        
    # Get forecast and confidence interval for steps ahead in future
    future = model.get_forecast(steps=steps, dynamic=True)
    future_conf = future.conf_int()

    # Plot future forecast with confidence interval
    if plot:
        ax = ts['2006':].plot(label='Observed', figsize=(10, 8))
        future.predicted_mean.plot(ax=ax, label='Forecast')
        ax.fill_between(future_conf.index,
                        future_conf.iloc[:, 0],
                        future_conf.iloc[:, 1], color='g', alpha=.3)
        ax.fill_betweenx(ax.get_ylim(), 
                         pd.to_datetime('2026-12-31'), 
                         predicted.index[-1], alpha=.1, zorder=-1)
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Crimes')
        plt.title('Future Forecast')
        ax.legend()
        plt.show()
        
    # show prediction for end of step-period 
    forecast = future.predicted_mean[-1]
    upper = future_conf.iloc[-1,1]
    lower = future_conf.iloc[-1,0]
    predictions = {}
    predictions['Upper Bound'] = upper
    predictions['Expected Forecast'] = forecast
    predictions['Lower Bound'] = lower
    predictions = pd.DataFrame.from_dict(predictions, orient='index', columns=['Prediction'])
  
    # calculate return percentages
    crime_2019 = ts.loc['2019-12-31']
    forecast_2026 = forecast
    forecast_lower = lower
    forecast_upper = upper
    return_percentage_predictions = {}
    predicted_percent_change = ((forecast_2026- crime_2019) / crime_2019)*100
    upper_percent_change = ((forecast_upper - crime_2019) / crime_2019)*100
    lower_percent_change = ((forecast_lower - crime_2019) / crime_2019)*100
    return_percentage_predictions['Predicted % Change'] = predicted_percent_change
    return_percentage_predictions['Upper % Change'] = upper_percent_change
    return_percentage_predictions['Lower % Change'] = lower_percent_change
    return_percentage_predictions = pd.DataFrame.from_dict(return_percentage_predictions,orient='index')
    
    if show:
        print(predictions)
        
        print('\n' + f'The RMSE of our forecast is {round(rmse, 2)}' + '\n')
        
        print(return_percentage_predictions)
```

Here is Manhattan's crime prediction plot.

![](https://raw.githubusercontent.com/helenpham0229/Flatiron-Capstone-NYC-Crimes/master/images/sarimax_manhattan.png)

## Facebook Prophet
Facebook Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.

Here is the function I created to get prediction with 95% confidence interval, fit the model, plot predictions, and calculates RMSE and percentage change. Prophet also plot trend, yearly, weekly, and daily, seasonality components.
```
def prophet_model(ts):
    
    '''Input time series data to get prediction with 95% confidence interval, fit the model, plot predictions, calculate 
    percentage change and RMSE'''

    
    # train-test split
    train_data = ts.iloc[:len(ts)-36]
    test_data = ts.iloc[len(ts)-36:]
    
    # Set the uncertainty interval to 95% and yearly seasonality
    model = Prophet(interval_width=0.95, yearly_seasonality = True, weekly_seasonality=True, daily_seasonality=True)
    # Fit the timeseries to model
    model.fit(train_data)
    # Use make_future_dataframe() with a monthly frequency and periods = 84 
    future = model.make_future_dataframe(periods=120, freq='M')
    # Predict the values for future dates and take the head of forecast
    forecast = model.predict(future)
    # Use Prophet's plot method to plot the predictions
    model.plot(forecast, uncertainty=True)
    plt.show()
    # Plot model components 
    model.plot_components(forecast)
    plt.show()
    # Calculate RMSE
    y_true = ts['y'].values
    y_pred = forecast['yhat'][:168].values
    RMSE = np.sqrt(mean_squared_error(y_true,y_pred))
    
    # show prediction for end of step-period 
    forecast_2026 = forecast['yhat'][251]
    forecast_lower = forecast['yhat_lower'][251]
    forecast_upper = forecast['yhat_upper'][251]
    print(f'Predicted Number of Crimes: {round(forecast_2026, 2)}' + '\n')
    print(f'Upper Number of Crimes: {round(forecast_upper, 2)}' + '\n')
    print(f'Lower Number of Crimes: {round(forecast_lower, 2)}' + '\n')
  
    # calculate return percentages
    crime_2019 = ts['y'][167]
    forecast_2026 = forecast['yhat'][251]
    forecast_lower = forecast['yhat_lower'][251]
    forecast_upper = forecast['yhat_upper'][251]
    return_percentage_predictions = {}
    predicted_percent_change = ((forecast_2026- crime_2019) / crime_2019)*100
    upper_percent_change = ((forecast_upper - crime_2019) / crime_2019)*100
    lower_percent_change = ((forecast_lower - crime_2019) / crime_2019)*100
    return_percentage_predictions['Predicted % Change'] = predicted_percent_change
    return_percentage_predictions['Upper % Change'] = upper_percent_change
    return_percentage_predictions['Lower % Change'] = lower_percent_change
    print(f'The RMSE of our forecast is {round(RMSE, 2)}' + '\n')
    print(f'Predicted % Change: {round(predicted_percent_change, 2)}' + '\n')
    print(f'Upper % Change: {round(upper_percent_change, 2)}' + '\n')
    print(f'Lower % Change: {round(lower_percent_change, 2)}' + '\n')
```

Manhattan's prediction plot should look similar to the one below. 

![](https://raw.githubusercontent.com/helenpham0229/Flatiron-Capstone-NYC-Crimes/master/images/prophet_manhattan.png)

## Holt-Winters Exponential Smoothing
Exponential smoothing assigns exponentially decreasing weights and values against historical data to decrease the value of the weight for the older data. It is used for forecasting time series data that exhibits both a trend and a seasonal variation.

Similar to the models above, I also created a function to train and test the time series. It also plot past data and future predictions as well as calculates RMSE and percentage change.

```
def hwes (ts, trend, seasonal):
    
    '''Enter time series data frame, trend (mul or add), and seasonal (mul or add) to train and test Holt-Winters Exponential
    Smoothing. This function also plots future predictions and calculate percentage change & RMSE'''
    
    # split between the training and the test data sets. The last 36 periods form the test data
    ts_train = ts.iloc[:-36]
    ts_test = ts.iloc[-36:]
    
    #build and train the model on the training data
    hwmodel = ExponentialSmoothing(ts_train,trend=trend, seasonal=seasonal, seasonal_periods=12).fit()

    #create an out of sample forcast for the next 120 steps beyond the final data point in the training data set
    test_pred = hwmodel.forecast(steps=120)
    test_pred.to_frame()

    #plot the training data, the test data and the forecast on the same plot
    ts_train['Crime_Number'].plot(legend=True, label='Train', figsize=(10,6))
    ts_test['Crime_Number'].plot(legend=True, label='Test')
    test_pred.plot(legend=True, label='Predicted')
    
    # calculate RMSE
    RMSE = np.sqrt(mean_squared_error(ts_test,test_pred[:36]))
    print('\n' + f'The RMSE of our forecast is {round(RMSE, 2)}' + '\n')

    # Number of crime by end of 2026
    crime_2026 = test_pred.iloc[-1]
    print(f'Predicted Number of Crimes by End of 2026 is {round(crime_2026, 2)}' + '\n')

    # Percent change
    crime_2019 = ts_test['Crime_Number'][-1]
    return_percentage_predictions = {}
    predicted_percent_change = ((crime_2026- crime_2019) / crime_2019)*100
    return_percentage_predictions['Predicted % Change'] = predicted_percent_change
    print(f'Predicted % Change: {round(predicted_percent_change, 2)}' + '\n')
```

And here is the prediction plot for Manhattan's number of crimes.
![](https://raw.githubusercontent.com/helenpham0229/Flatiron-Capstone-NYC-Crimes/master/images/hwes_manhattan.png)

## Conclusion
For this particular forecasting project, I found that SARIMAX returned the lowest RMSE. However, I think that each method has its own advantage. SARIMAX handles seasonality and component much better but it takes much longer to find the hyperparameters and the best model.  Holt-Winters Exponential Smoothing is the fastest model to train. And Facebook Prophet clearly shows seasonality components (yearly, weekly, and daily) and "outliers".

Give these models a try and let me know which one you like :) 
