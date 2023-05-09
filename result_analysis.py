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

file_path1 = './data/result/df_final_sarima_1.csv'
file_path2 = './data/result/df_final_sarima_2.csv'
file_path3 = './data/result/df_final_randomforest_1.csv'
file_path4 = './data/result/df_final_randomforest_2.csv'
file_path5 = './data/result/df_final_xgboost_1.csv'
file_path6 = './data/result/df_final_xgboost_2.csv'
file_path7 = './data/result/df_lstm_1_temp.csv'
file_path8 = './data/result/df_lstm_2_temp.csv'

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))*100

if __name__ == "__main__":
    df_temp = pd.read_csv(file_path1)
    product_temp_list = df_temp['Description'].unique()
    df_temp = df_temp[~df_temp['predict_Quantity'].isna()]

    df = pd.read_csv(file_path5)
    df = df[~df['predict_Quantity'].isna()]

    # 예측 해야 하는 값이 0인 것을 제외 하고, 성능 비교
    df = df[df['Quantity'] >= 1]
    df = df[df['Description'].isin(product_temp_list)]

    # MAE 구하기
    mae_total = np.round(mean_absolute_error(df['Quantity'], df['predict_Quantity']),2)
    mae_Erratic = np.round(mean_absolute_error(df[df['Erratic'] == 1]['Quantity'], df[df['Erratic'] == 1]['predict_Quantity']),2)
    mae_Intermittent = np.round(mean_absolute_error(df[df['Intermittent'] == 1]['Quantity'],df[df['Intermittent'] == 1]['predict_Quantity']),2)
    mae_Lumpy = np.round(mean_absolute_error(df[df['Lumpy'] == 1]['Quantity'], df[df['Lumpy'] == 1]['predict_Quantity']),2)
    mae_Smooth = np.round(mean_absolute_error(df[df['Smooth'] == 1]['Quantity'], df[df['Smooth'] == 1]['predict_Quantity']),2)

    # RMSE 구하기
    rmse_total = np.round(np.sqrt(mean_squared_error(df['Quantity'], df['predict_Quantity'])),2)
    rmse_Erratic = np.round(np.sqrt(mean_squared_error(df[df['Erratic'] == 1]['Quantity'], df[df['Erratic'] == 1]['predict_Quantity'])),2)
    rmse_Intermittent = np.round(np.sqrt(mean_squared_error(df[df['Intermittent'] == 1]['Quantity'], df[df['Intermittent'] == 1]['predict_Quantity'])),2)
    rmse_Lumpy = np.round(np.sqrt(mean_squared_error(df[df['Lumpy'] == 1]['Quantity'], df[df['Lumpy'] == 1]['predict_Quantity'])),2)
    rmse_Smooth = np.round(np.sqrt(mean_squared_error(df[df['Smooth'] == 1]['Quantity'], df[df['Smooth'] == 1]['predict_Quantity'])),2)

    # R2 스코어 구하기
    r2s_total = np.round(r2_score(df['Quantity'], df['predict_Quantity']),4)
    r2s_Erratic = np.round(r2_score(df[df['Erratic'] == 1]['Quantity'], df[df['Erratic'] == 1]['predict_Quantity']),4)
    r2s_Intermittent = np.round(r2_score(df[df['Intermittent'] == 1]['Quantity'],df[df['Intermittent'] == 1]['predict_Quantity']),4)
    r2s_Lumpy = np.round(r2_score(df[df['Lumpy'] == 1]['Quantity'], df[df['Lumpy'] == 1]['predict_Quantity']),4)
    r2s_Smooth = np.round(r2_score(df[df['Smooth'] == 1]['Quantity'], df[df['Smooth'] == 1]['predict_Quantity']),4)

    # MAPE 구하기
    mape_total = np.round(mean_absolute_percentage_error(df['Quantity'], df['predict_Quantity']),2)
    mape_Erratic = np.round(mean_absolute_percentage_error(df[df['Erratic'] == 1]['Quantity'], df[df['Erratic'] == 1]['predict_Quantity']),2)
    mape_Intermittent = np.round(mean_absolute_percentage_error(df[df['Intermittent'] == 1]['Quantity'], df[df['Intermittent'] == 1]['predict_Quantity']),2)
    mape_Lumpy = np.round(mean_absolute_percentage_error(df[df['Lumpy'] == 1]['Quantity'], df[df['Lumpy'] == 1]['predict_Quantity']),2)
    mape_Smooth = np.round(mean_absolute_percentage_error(df[df['Smooth'] == 1]['Quantity'], df[df['Smooth'] == 1]['predict_Quantity']),2)

    # 예측 건수 확인
    final_product_cnt = np.round(len(df['Description'].unique()),2)
    final_product_Erratic_cnt = np.round(len(df[df['Erratic'] == 1]['Description'].unique()),2)
    final_product_Intermittent_cnt = np.round(len(df[df['Intermittent'] == 1]['Description'].unique()),2)
    final_product_Lumpy_cnt = np.round(len(df[df['Lumpy'] == 1]['Description'].unique()),2)
    final_product_Smooth_cnt = np.round(len(df[df['Smooth'] == 1]['Description'].unique()),2)