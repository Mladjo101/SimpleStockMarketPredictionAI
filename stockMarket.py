import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import Input

# Učitavanje podataka o cijeni dionica
def load_data(stock, startDate, endDate):
    data = yf.download(stock, start=startDate, end=endDate)
    return data['Close'].values.reshape(-1, 1)

# Priprema podataka za obradu
def preprocess_data(data, trainSize=0.8):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaledData = scaler.fit_transform(data)
    
    trainDataLen = int(len(data) * trainSize)
    
    trainData = scaledData[:trainDataLen]
    testData = scaledData[trainDataLen:]
    
    return trainData, testData, scaler

# Kreiranje nizova za LSTM model
def create_sequences(data, seqLength):
    featureValues, targetValues = [], []
    for i in range(len(data) - seqLength):
        featureValues.append(data[i:i+seqLength, 0])
        targetValues.append(data[i+seqLength, 0])
    return np.array(featureValues), np.array(targetValues)

# Kreiranje LSTM modela
def build_model():
    model = Sequential()
    model.add(Input(shape=(60, 1))) 
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Treniranje modela, te predikcija i prikaz rezultata
def train_and_evaluate(stock, startDate, endDate):

    data = load_data(stock, startDate, endDate)
    
    trainData, testData, scaler = preprocess_data(data)
    
    seqLength = 60
    featureValuesTrain, targetValuesTrain = create_sequences(trainData, seqLength)
    featureValuesTest, targetValues_test = create_sequences(testData, seqLength)
    
    featureValuesTrain = featureValuesTrain.reshape(featureValuesTrain.shape[0], featureValuesTrain.shape[1], 1)
    featureValuesTest = featureValuesTest.reshape(featureValuesTest.shape[0], featureValuesTest.shape[1], 1)
    
    model = build_model()
    
    model.fit(featureValuesTrain, targetValuesTrain, batch_size=1, epochs=2)
    
    predictions = model.predict(featureValuesTest)
    predictions = scaler.inverse_transform(predictions)
    
    trainDataLen = len(trainData)
    real_prices = data[trainDataLen + seqLength:]  
    testRange = np.arange(trainDataLen + seqLength, trainDataLen + seqLength + len(predictions))
    
    plt.figure(figsize=(10, 6))
    plt.plot(data, color='blue', label='Actual Stock Price') 
    plt.plot(testRange, predictions, color='red', label='Predicted Stock Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Pokretanje modela (dionica, datum od, datum do)
train_and_evaluate('TSLA', '2015-01-01', '2022-01-01')