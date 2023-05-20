import pandas as pd

pd.set_option('display.max_columns', None)
import numpy as np
import datetime

# plot packages
import matplotlib.pyplot as plt

# metrics for model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error

# file_path1 = './data/result/df_final_xgboost_1_lookback_2.csv'
file_path2 = './data/result/df_final_xgboost_2_lookback_2.csv'
# file_path3 = './data/result/df_final_xgboost_1_lookback_3.csv'
file_path4 = './data/result/df_final_xgboost_2_lookback_3.csv'
# file_path5 = './data/result/df_final_xgboost_1_lookback_4.csv'
file_path6 = './data/result/df_final_xgboost_2_lookback_4.csv'
# file_path7 = './data/result/df_final_xgboost_1_lookback_5.csv'
file_path8 = './data/result/df_final_xgboost_2_lookback_5.csv'
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.abs(y_true))*100)

if __name__ == "__main__":
    file_list = [file_path2 , file_path4 , file_path6 , file_path8 ]
    product_list = []
    product_temp = []

    for f in file_list:
        df_temp = pd.read_csv(f)
        df_temp = df_temp[~df_temp['predict_Quantity'].isna()]
        df_temp = df_temp[df_temp['Quantity'] >= 1]
        df_temp = df_temp[df_temp['predict_Quantity'] >= 1]
        product_list = df_temp['Description'].unique()
        if len(product_temp) == 0:
            product_temp = product_list
            print('inin1', len(product_temp))
        else:
            product_temp = list(set(product_temp) & set(product_list))
            print('inin2',len(product_temp))

    df = pd.read_csv(file_path4)
    # df = df[df['Description'].isin(product_temp)]
    print(df)


    # # 예측 해야 하는 값이 0인 것을 제외 하고, 성능 비교
    # df = df[df['Quantity'] >= 1]
    # df = df[df['predict_Quantity'] >= 0]

    pattern = 'Intermittent'
    # MAE 구하기
    mae = np.round(mean_absolute_error(df['Quantity'],df['predict_Quantity']), 2)
    mae_1 = np.round(mean_absolute_error(df[(df[pattern] == 1) & (df['product_cluster'] == 0)]['Quantity'], df[(df[pattern] == 1) & (df['product_cluster'] == 0)]['predict_Quantity']),2)
    mae_2 = np.round(mean_absolute_error(df[(df[pattern] == 1) & (df['product_cluster'] == 1)]['Quantity'], df[(df[pattern] == 1) & (df['product_cluster'] == 1)]['predict_Quantity']),2)
    mae_3 = np.round(mean_absolute_error(df[(df[pattern] == 1) & (df['product_cluster'] == 2)]['Quantity'], df[(df[pattern] == 1) & (df['product_cluster'] == 2)]['predict_Quantity']),2)
    mae_4 = np.round(mean_absolute_error(df[(df[pattern] == 1) & (df['product_cluster'] == 3)]['Quantity'], df[(df[pattern] == 1) & (df['product_cluster'] == 3)]['predict_Quantity']),2)
    mae_5 = np.round(mean_absolute_error(df[(df[pattern] == 1) & (df['product_cluster'] == 4)]['Quantity'], df[(df[pattern] == 1) & (df['product_cluster'] == 4)]['predict_Quantity']),2)

    # MAPE 구하기
    mape = np.round(mean_absolute_percentage_error(df['Quantity'], df['predict_Quantity']), 2)
    mape_1 = np.round(mean_absolute_percentage_error(df[(df[pattern] == 1) & (df['product_cluster'] == 0)]['Quantity'],df[(df[pattern] == 1) & (df['product_cluster'] == 0)]['predict_Quantity']), 2)
    mape_2 = np.round(mean_absolute_percentage_error(df[(df[pattern] == 1) & (df['product_cluster'] == 1)]['Quantity'],df[(df[pattern] == 1) & (df['product_cluster'] == 1)]['predict_Quantity']), 2)
    mape_3 = np.round(mean_absolute_percentage_error(df[(df[pattern] == 1) & (df['product_cluster'] == 2)]['Quantity'],df[(df[pattern] == 1) & (df['product_cluster'] == 2)]['predict_Quantity']), 2)
    mape_4 = np.round(mean_absolute_percentage_error(df[(df[pattern] == 1) & (df['product_cluster'] == 3)]['Quantity'],df[(df[pattern] == 1) & (df['product_cluster'] == 3)]['predict_Quantity']), 2)
    mape_5 = np.round(mean_absolute_percentage_error(df[(df[pattern] == 1) & (df['product_cluster'] == 4)]['Quantity'],df[(df[pattern] == 1) & (df['product_cluster'] == 4)]['predict_Quantity']), 2)

    # RMSE 구하기
    rmse = np.round(np.sqrt(mean_squared_error(df['Quantity'], df['predict_Quantity'])), 2)
    rmse_1 = np.round(np.sqrt(mean_squared_error(df[(df[pattern] == 1) & (df['product_cluster'] == 0)]['Quantity'],df[(df[pattern] == 1) & (df['product_cluster'] == 0)]['predict_Quantity'])), 2)
    rmse_2 = np.round(np.sqrt(mean_squared_error(df[(df[pattern] == 1) & (df['product_cluster'] == 1)]['Quantity'],df[(df[pattern] == 1) & (df['product_cluster'] == 1)]['predict_Quantity'])), 2)
    rmse_3 = np.round(np.sqrt(mean_squared_error(df[(df[pattern] == 1) & (df['product_cluster'] == 2)]['Quantity'],df[(df[pattern] == 1) & (df['product_cluster'] == 2)]['predict_Quantity'])), 2)
    rmse_4 = np.round(np.sqrt(mean_squared_error(df[(df[pattern] == 1) & (df['product_cluster'] == 3)]['Quantity'],df[(df[pattern] == 1) & (df['product_cluster'] == 3)]['predict_Quantity'])), 2)
    rmse_5 = np.round(np.sqrt(mean_squared_error(df[(df[pattern] == 1) & (df['product_cluster'] == 4)]['Quantity'],df[(df[pattern] == 1) & (df['product_cluster'] == 4)]['predict_Quantity'])), 2)

    # 예측 건수 확인
    final_product_cnt = len(df['Description'].unique())
    final_product_Erratic_cnt = len(df[df['Erratic'] == 1]['Description'].unique())
    final_product_Intermittent_cnt = len(df[df['Intermittent'] == 1]['Description'].unique())
    final_product_Lumpy_cnt = len(df[df['Lumpy'] == 1]['Description'].unique())
    final_product_Smooth_cnt = len(df[df['Smooth'] == 1]['Description'].unique())

    pattern = 'Smooth'
    cluster = 4
    final_product_Erratic_cluster_cnt = len(df[(df[pattern] == 1) & (df['product_cluster'] == cluster)]['Description'].unique())
