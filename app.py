import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import pandas_datareader as data
from keras.models import load_model
import yfinance as yf
import streamlit as st



start = '2010-01-01'
end = '2024-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter a Stock Ticker ','AAPL')
# Use yfinance to download data instead of pandas_datareader
df = yf.download(user_input, start=start, end=end) 
df.columns = [ 'Close', 'High', 'Low', 'Open', 'Volume']



st.subheader('Data from 2010 t0 2024')
st.write(df.describe())

#Visualization

st.subheader('Closing Price Vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, color = 'g')
st.pyplot(fig)


ma100  = df.Close.rolling(100).mean()
st.subheader('Closing Price Vs Time Chart with 100 Mean Average')
fig1 = plt.figure(figsize=(12,6))
plt.plot(ma100, color = 'black')
plt.plot(df.Close, color = 'b')
st.pyplot(fig1)

ma200  = df.Close.rolling(200).mean()
st.subheader('Closing Price Vs Time Chart with 100 & 200 Mean Average')
fig2 = plt.figure(figsize=(12,6))
plt.plot(ma200, color = 'g')
plt.plot(ma100, color = 'r')
plt.plot(df.Close, color = 'b')
st.pyplot(fig2)


#splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn .preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


# Load the model 
model = load_model('keras_model.h5')

#Testing part
past_100_days = data_training.tail(100)

final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])


x_test, y_test = np.array(x_test),np.array(y_test)

y_predicted = model.predict(x_test)

scaler.scale_

scale_factor = 1/scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


#Final Graph 
st.subheader('Plot for Original and Predicted Graph ')
fig3 = plt.figure(figsize=(12,6))
plt.plot(y_test,'g',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)