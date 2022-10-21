# importing required libraries
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


def test_stationarity(timeseries):
    # Determing rolling statistics
    # rolmean = pd.rolling_mean(timeseries, window=12)
    rolmean = timeseries.rolling(12).mean()
    # rolstd = pd.rolling_std(timeseries, window=12)
    rolstd = timeseries.rolling(12).std()

    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


# Now, we will load the data set and look at some initial rows and data types of the columns:
data = pd.read_csv('AirPassengers.csv')
print(data.head())
print('\n Data Types:')
print(data.dtypes)

# The data contains a particular month and number of passengers
# travelling in that month. In order to read the data as a time series,
# we have to pass special arguments to the read_csv command:
# dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
dateparse = lambda dates: datetime.strptime(dates, '%Y-%m')
data = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month', date_parser=dateparse)
print('\n Parsed Data:')
print(data.head())

# NOTE: You can run remaining codes in this article as well, using this live coding window.
ts = data['#Passengers']
ts.head(10)

#1. Specific the index as a string constant:
print(ts['1949-01-01'])

#2. Import the datetime library and use 'datetime' function:
print(ts[datetime(1949, 1, 1)])

#1. Specify the entire range:
print(ts['1949-01-01':'1949-05-01'])

#2. Use ':' if one of the indices is at ends:
print(ts[:'1949-05-01'])

print(ts['1949'])
plt.plot(ts)
plt.show()
print('ok')

# test_stationarity(ts)

ts_log = np.log(ts)
moving_avg = ts_log.rolling(12).mean()
plt.plot(ts_log)
plt.plot(moving_avg, color='red')
plt.show()

ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(12)
ts_log_moving_avg_diff.dropna(inplace=True)
# test_stationarity(ts_log_moving_avg_diff)

# expwighted_avg = pd.ewma(ts_log, halflife=12)
expwighted_avg = ts_log.ewm(ignore_na=False,span=30.007751938,min_periods=12,adjust=True).mean()
plt.plot(ts_log)
# plt.plot(expwighted_avg, color='red')
# plt.show()
ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)
