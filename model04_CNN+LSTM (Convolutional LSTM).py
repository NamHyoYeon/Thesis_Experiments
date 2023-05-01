import pandas as pd
import numpy as np
import datetime

# plot packages
import matplotlib.pyplot as plt

# model packages
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import ConvLSTM2D, Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# metrics for model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

file_path = './data/df_final_with_bf1mm.csv'

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    df = pd.read_csv(file_path)

    # 전월 값이 있는 ROW 부터 모델에 사용
    df = df[~df['Quantity_bf1mm'].isna()]
    # float -> int
    df[['Quantity_bf1mm','0_cluster_bf1mm','1_cluster_bf1mm','2_cluster_bf1mm','3_cluster_bf1mm','4_cluster_bf1mm']] = df[['Quantity_bf1mm','0_cluster_bf1mm','1_cluster_bf1mm','2_cluster_bf1mm','3_cluster_bf1mm','4_cluster_bf1mm']].astype(int)
    df = df.set_index(['YYYYMM'])
    df['predict_Quantity'] = np.nan
    print(df.head())
    df.sort_values(['Description','YYYYMM'], inplace=True)

    product_list = df['Description'].unique()
    bf = datetime.datetime.now()

    print(df.head())
    print(len(df[df['Description'] == product_list[1]]))

    # test code
    df_test = df[df['Description'] == product_list[1]]
    index_list = df_test.index.to_list()
    max = index_list[-1]
    second = index_list[-2]
    third = index_list[-3]

    train_data = df_test[:third]
    val_data = df_test[second:second]
    test_data = df_test[max:]

    print(train_data)
    print(val_data)
    print(test_data)

    timestamp = len(df_test)
    print(timestamp)

    # Define the model architecture
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1, 3), activation='relu', input_shape=(None, rows, columns, features)))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=10, batch_size=64)

    # Evaluate the model
    score = model.evaluate(test_data, test_labels, batch_size=64)

    # Make predictions
    predictions = model.predict(new_data)


    af = datetime.datetime.now