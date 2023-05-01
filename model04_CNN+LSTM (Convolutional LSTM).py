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
    df_test = df[df['Description'] == product_list[0]]
    index_list = df_test.index.to_list
    second = np.argsort(index_list)[-2]
    third = np.argsort(index_list)[-3]
    print(index_list)
    print(index_list[second])

    max_index = df_test.index.max()
    second_index = index_list[second]
    third_index = index_list[third]

    train_data = df_test[:third_index]
    val_data = df_test[second_index:second_index]
    test_data = df_test[max_index:]

    print(val_data)
    print(test_data)

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