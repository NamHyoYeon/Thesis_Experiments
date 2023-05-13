import pandas as pd
pd.set_option('display.max_columns', None)
from pycaret.regression import *
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

file_path = './data/df_final_with_bf1mm_2.csv'

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

    print(product_list[0])
    df_tgt = df[df['Description'] == product_list[1]]

    if len(df_tgt) >= 10:
        df_tgt = df_tgt[
            ['Description', 'Quantity', '0_cluster', '1_cluster', '2_cluster', '3_cluster', '4_cluster']]
        index_list = df_tgt.index.to_list()
        max_index = index_list[-1]

        df_input = pd.DataFrame()
        look_back = 5

        for i in range(len(index_list)):
            bf_index = 1
            df_temp = df_tgt[(df_tgt.index == index_list[i]) & (df_tgt['Description'] == product_list[0])]
            if i >= look_back:
                for j in range(i - look_back, i):
                    temp = df_tgt[(df_tgt.index == index_list[j]) & (df_tgt['Description'] == product_list[0])]
                    temp = temp.reset_index()
                    temp.index = [index_list[i]]
                    temp = temp[['Quantity', '0_cluster', '1_cluster', '2_cluster', '3_cluster', '4_cluster']]
                    temp = temp.rename(columns=lambda x: f'bf_{bf_index}_{x}')

                    df_temp = pd.concat([df_temp, temp], axis=1)
                    if j == i - 1:
                        df_input = pd.concat([df_input, df_temp], axis=0)
                    bf_index = bf_index + 1

        target_variable = 'Quantity'
        # 당월 값d은 및 상품명은 제외
        not_features = ['Description', '0_cluster', '1_cluster', '2_cluster', '3_cluster', '4_cluster']
        tgt_scale = [x for x in df_input.columns if x not in not_features]
        features = [x for x in df_input.columns if (x != target_variable) & (x not in not_features)]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data_x = scaler.fit_transform(df_input[features])

        x_train = scaled_data_x[:-1, 1:]
        x_test = scaled_data_x[-1:, 1:]

        print(x_train)
        df_1 = pd.DataFrame(x_train)
        df_3 = pd.DataFrame(x_test)
        df_1.reset_index(inplace=True)
        print(len(df_1))
        df_2 = pd.DataFrame(y_train, columns=['Quantity'])
        df_2.reset_index(inplace=True)
        print(df_2)

        df_tt = pd.concat([df_1, df_2], axis=1)
        print(df_tt.info())

        print(df_1)
        del df_tt['index']

        print(y_train)
        df_1.columns = features

        y_train = df_input.loc[:index_list[-2], target_variable]
        y_test = df_input.loc[index_list[-1]:, target_variable]

        # Initialize the regression setup
        regression_setup = setup(df_tt, target='Quantity')
        best = compare_models(sort='RMSE')
        print(best)
        # Compare different regression models and select the best one
        comp = compare_models(sort = 'RMSE')
        xgb = create_model('xgboost', cross_validation=False)
        tuned_xgb = tune_model(xgb, optimize='RMSE', n_iter=5)

        final_model = finalize_model(xgb)
        pred = predict_model(final_model, data=df_3)

        print(pred)
        print(y_test)

        models()

        print(best_model)

        xgb = XGBRegressor(objective='reg:squarederror', n_estimators=5, learning_rate=0.01, random_state=42,
                           max_depth=3)
        xgb.fit(x_train, y_train)
        y_pred = xgb.predict(x_test)

        print(y_pred)
        print(y_test)

        df.loc[(df['Description'] == p) & (df.index == max_index), 'predict_Quantity'] = int(np.ceil(y_pred[0]))
