# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from pyramid.arima import auto_arima
import pyramid as pm

# reading the data
df = pd.read_csv('NSE-TATAGLOBAL.csv')

# looking at the first five rows of the data
print(df.head())
print('\n Shape of the data:')
print(df.shape)

#setting index as date values
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']


data = df.sort_index(ascending=True, axis=0)

# splitting into train and validation
train = data[:1800]
valid = data[1800:]

training = train['Close']
validation = valid['Close']

# model = auto_arima(training, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
model = pm.auto_arima()
model = pm.auto_arima(training, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
model.fit(training)

forecast = model.predict(n_periods=248)
forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])