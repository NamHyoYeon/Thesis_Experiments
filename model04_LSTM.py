import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import datetime

# plot packages
import matplotlib.pyplot as plt

# model packages
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# metrics for model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

file_path = './data/df_final_with_bf1mm.csv'

if __name__ == "__main__":
    df = pd.read_csv(file_path)
    df = df[['YYYYMM','Description','Quantity','0_cluster','1_cluster','2_cluster','3_cluster','4_cluster','Erratic','Intermittent','Lumpy','Smooth']]
    df.set_index('YYYYMM', inplace=True)
    print(df.head())

    product_list = df['Description'].unique()

    ###### test code ######
    df_test = df[df['Description'] == product_list[0]]
    df_test.sort_values(['YYYYMM'], inplace=True)
    print(df_test)

    max_yyyymm = df_test.index.max()
    print(max_yyyymm)
    length = len(df_test)

    # Min-Max Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_test[['Quantity', '0_cluster', '1_cluster', '2_cluster', '3_cluster', '4_cluster']])

    print(scaled_data.shape)
    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        print(len(dataset))
        for i in range(len(dataset) - look_back):
            print(dataset[i:(i + look_back), :])
            print(dataset[i + look_back, :])

            dataX.append(dataset[i:(i + look_back), :])
            dataY.append(dataset[i + look_back, :])
        return np.array(dataX), np.array(dataY)

    # train, test 데이터 만들기
    train_data = scaled_data[:-1, :]
    test_data = scaled_data[-1:, :]

    look_back = 2
    X_train, y_train = create_dataset(train_data, look_back)
    X_test, y_test = create_dataset(test_data, look_back)

    print(X_train.shape)
    print(y_train.shape)

    # reshape, 차원 잘 맞추기 !! 어떻게 맞추어야 하나??
    X_train = np.reshape(X_train, (6, 20, 1))
    print(X_train.shape[1])

    # 모델 생성
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(20, 1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 모델 학습
    model.fit(X_train, y_train, epochs=100, batch_size=1)












    # 다음 값 예측
    train_predict = model.predict(X_train)
    print(train_predict.shape)
    print(train_predict.reshape(1,10))
    train_predict = scaler.inverse_transform(train_predict.reshape(1,6,10))
    print(train_predict[0][1])

    print(train_predict.reshape(1,10))
    print(df_test['Quantity'])

    y_train = scaler.inverse_transform(y_train.reshape(-1, 1))