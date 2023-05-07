import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import datetime

# plot packages
# import matplotlib.pyplot as plt

# model packages
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

from tensorflow.python.client import device_lib
import tensorflow as tf
tf.config.list_physical_devices('GPU')

print(device_lib.list_local_devices())
import tensorflow as tf
from keras import backend as K
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
K.set_session(sess)

# metrics for model
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import r2_score
#from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

file_path = './data/df_final_with_bf1mm.csv'

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    print(len(dataset))
    for i in range(len(dataset) - look_back):
        print(i)
        print(dataset[i:(i + look_back), :])
        print(dataset[i + look_back, :])

        dataX.append(dataset[i:(i + look_back), :])
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)

if __name__ == "__main__":
    df = pd.read_csv(file_path)
    df = df[['YYYYMM','Description','Quantity','0_cluster','1_cluster','2_cluster','3_cluster','4_cluster','Erratic','Intermittent','Lumpy','Smooth']]
    df.set_index('YYYYMM', inplace=True)
    print(df.head())

    product_list = df['Description'].unique()
    df['predict_Quantity'] = np.nan
    bf = datetime.datetime.now()

    for p in product_list:
        print(p)
        df_tgt = df[df['Description'] == p]
        if len(df_tgt) > 8:
            df_tgt.sort_values(['YYYYMM'], inplace=True)

            # 상품별 max 월 찾기
            max_yyyymm = df_tgt.index.max()
            print(max_yyyymm)
            # 상품별 time stamp 기간
            length = len(df_tgt)

            # LSTM 모델 Features 사용을 위한 Min-Max Scaling
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df_tgt[['Quantity', '0_cluster', '1_cluster', '2_cluster', '3_cluster', '4_cluster']])
            print(scaled_data.shape)

            # train, test 데이터 만들기
            train_data = scaled_data[:-1, :]
            test_data = scaled_data[-1:, :]

            look_back = 2
            X_train, y_train = create_dataset(train_data, look_back)

            # 모델 생성
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dense(units=1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            # 모델 학습
            model.fit(X_train, y_train, epochs=100, batch_size=1)

            # 다음 값 예측
            train_predict = model.predict(X_train)
            print(train_predict)
            predict_1_mm = train_predict[0].reshape(1,3)
            predict_1_mm = scaler.inverse_transform(predict_1_mm)
            print(int(np.ceil(predict_1_mm[0][0])))

            df.loc[(df['Description'] == p) & (df.index == max_yyyymm), 'predict_Quantity'] = int(np.ceil(predict_1_mm[0][0]))
            
            
        print(df[~df['predict_Quantity'].isna()][['Quantity','predict_Quantity']])

        df_val = df[~df['predict_Quantity'].isna()][['Quantity','predict_Quantity','Erratic','Intermittent','Lumpy','Smooth']]

        mae2_total = mean_absolute_error(df_val['Quantity'], df_val['predict_Quantity'])
        mae2_Erratic = mean_absolute_error(df_val[df_val['Erratic'] == 1]['Quantity'],df_val[df_val['Erratic'] == 1]['predict_Quantity'])
        mae2_Intermittent = mean_absolute_error(df_val[df_val['Intermittent'] == 1]['Quantity'],df_val[df_val['Intermittent'] == 1]['predict_Quantity'])
        mae2_Lumpy = mean_absolute_error(df_val[df_val['Lumpy'] == 1]['Quantity'],df_val[df_val['Lumpy'] == 1]['predict_Quantity'])
        mae2_Smooth = mean_absolute_error(df_val[df_val['Smooth'] == 1]['Quantity'],df_val[df_val['Smooth'] == 1]['predict_Quantity'])

    af = datetime.datetime.now()