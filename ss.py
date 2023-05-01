import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import ConvLSTM2D, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
%matplotlib inline

# Load the data
data = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/train.csv', parse_dates=['Date'])
data = data[data['Store'] == 1]
data = data[['Date', 'Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']].dropna()
data = data.resample('M', on='Date').sum()
data.drop('Store', axis=1, inplace=True)

# Scale the data
scaler = MinMaxScaler(feature_range=(-1, 1))
data = scaler.fit_transform(data)

# Prepare the data for training
timesteps = 12
X = []
y = []
for i in range(timesteps, len(data)):
    X.append(data[i-timesteps:i])
    y.append(data[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], 1, 1, timesteps, 5))

# Split the data into training and validation sets
train_size = int(len(X) * 0.75)
val_size = len(X) - train_size
train_data, val_data = X[:train_size], X[train_size:]
train_labels, val_labels = y[:train_size], y[train_size:]

# Define the model architecture
model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(1, 1, timesteps, 5)))
model.add(Flatten())
model.add(Dense(units=1))

# Compile the model
model.compile(loss='mean_squared_error', optimizer=Adam())

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=20, batch_size=64, callbacks=[early_stopping])

# Evaluate the model
test_data = np.reshape(scaler.transform(pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/test.csv', parse_dates=['Date'])['Weekly_Sales'].values.reshape(-1, 1)), (-1, 1, 1, timesteps, 5))
score = model.evaluate(test_data, np.zeros((len(test_data), 1)))

# Make predictions
predictions = scaler.inverse_transform(model.predict(test_data)).reshape(-1)

# Plot the results
plt.plot(pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/test.csv', parse_dates=['Date'])['Weekly_Sales'].values, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()
