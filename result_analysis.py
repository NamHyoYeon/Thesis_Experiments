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

file_path1 = './data/result/df_final_sarima_1.csv'
file_path2 = './data/result/df_final_sarima_2.csv'
file_path3 = './data/result/df_final_randomforest_1.csv'
file_path4 = './data/result/df_final_randomforest_2.csv'
file_path5 = './data/result/df_final_xgboost_1.csv'
file_path6 = './data/result/df_final_xgboost_2.csv'
file_path7 = './data/result/df_lstm_1_temp.csv'
file_path8 = './data/result/df_lstm_2_temp.csv'

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

if __name__ == "__main__":
    df = pd.read_csv(file_path8)
    df = df[~df['predict_Quantity'].isna()]

    # MAE 구하기
    mae_total = mean_absolute_error(df['Quantity'], df['predict_Quantity'])
    mae_Erratic = mean_absolute_error(df[df['Erratic'] == 1]['Quantity'], df[df['Erratic'] == 1]['predict_Quantity'])
    mae_Intermittent = mean_absolute_error(df[df['Intermittent'] == 1]['Quantity'],df[df['Intermittent'] == 1]['predict_Quantity'])
    mae_Lumpy = mean_absolute_error(df[df['Lumpy'] == 1]['Quantity'], df[df['Lumpy'] == 1]['predict_Quantity'])
    mae_Smooth = mean_absolute_error(df[df['Smooth'] == 1]['Quantity'], df[df['Smooth'] == 1]['predict_Quantity'])

    # RMSE 구하기
    rmse_total = np.sqrt(mean_squared_error(df['Quantity'], df['predict_Quantity']))
    rmse_Erratic = np.sqrt(mean_squared_error(df[df['Erratic'] == 1]['Quantity'], df[df['Erratic'] == 1]['predict_Quantity']))
    rmse_Intermittent = np.sqrt(mean_squared_error(df[df['Intermittent'] == 1]['Quantity'], df[df['Intermittent'] == 1]['predict_Quantity']))
    rmse_Lumpy = np.sqrt(mean_squared_error(df[df['Lumpy'] == 1]['Quantity'], df[df['Lumpy'] == 1]['predict_Quantity']))
    rmse_Smooth = np.sqrt(mean_squared_error(df[df['Smooth'] == 1]['Quantity'], df[df['Smooth'] == 1]['predict_Quantity']))

    # R2 스코어 구하기
    r2s_total = r2_score(df['Quantity'], df['predict_Quantity'])
    r2s_Erratic = r2_score(df[df['Erratic'] == 1]['Quantity'], df[df['Erratic'] == 1]['predict_Quantity'])
    r2s_Intermittent = r2_score(df[df['Intermittent'] == 1]['Quantity'],df[df['Intermittent'] == 1]['predict_Quantity'])
    r2s_Lumpy = r2_score(df[df['Lumpy'] == 1]['Quantity'], df[df['Lumpy'] == 1]['predict_Quantity'])
    r2s_Smooth = r2_score(df[df['Smooth'] == 1]['Quantity'], df[df['Smooth'] == 1]['predict_Quantity'])

