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

    df_product_cluster = pd.read_csv(r'./data/df_product_cluster.csv')

    df1 = pd.read_csv(file_path2)
    df1 = pd.merge(df1, df_product_cluster, on='Description' , how='left')

    df2 = pd.read_csv(file_path4)
    df2 = pd.merge(df2, df_product_cluster, on='Description' , how='left')

    df3 = pd.read_csv(file_path6)
    df3 = pd.merge(df3, df_product_cluster, on='Description' , how='left')

    df4 = pd.read_csv(file_path8)
    df4 = pd.merge(df4, df_product_cluster, on='Description' , how='left')

    df_final = pd.DataFrame()

    df_final = pd.concat([df_final, df1[(df1['Erratic'] == 1) & (df1['product_cluster'] == 0)]], axis=0)
    df_final = pd.concat([df_final, df1[(df1['Erratic'] == 1) & (df1['product_cluster'] == 2)]], axis=0)
    df_final = pd.concat([df_final, df1[(df1['Intermittent'] == 1) & (df1['product_cluster'] == 2)]], axis=0)
    df_final = pd.concat([df_final, df1[(df1['Lumpy'] == 1) & (df1['product_cluster'] == 2)]], axis=0)
    df_final = pd.concat([df_final, df1[(df1['Lumpy'] == 1) & (df1['product_cluster'] == 4)]], axis=0)
    df_final = pd.concat([df_final, df1[(df1['Smooth'] == 1) & (df1['product_cluster'] == 1)]], axis=0)
    df_final = pd.concat([df_final, df1[(df1['Smooth'] == 1) & (df1['product_cluster'] == 2)]], axis=0)

    df_final = pd.concat([df_final, df2[(df2['Lumpy'] == 1) & (df2['product_cluster'] == 0)]], axis=0)
    df_final = pd.concat([df_final, df2[(df2['Lumpy'] == 1) & (df2['product_cluster'] == 1)]], axis=0)

    df_final = pd.concat([df_final, df3[(df3['Erratic'] == 1) & (df3['product_cluster'] == 1)]], axis=0)
    df_final = pd.concat([df_final, df3[(df3['Lumpy'] == 1) & (df3['product_cluster'] == 3)]], axis=0)
    df_final = pd.concat([df_final, df3[(df3['Smooth'] == 1) & (df3['product_cluster'] == 0)]], axis=0)
    df_final = pd.concat([df_final, df3[(df3['Smooth'] == 1) & (df3['product_cluster'] == 4)]], axis=0)

    df_final = pd.concat([df_final, df4[(df4['Erratic'] == 1) & (df4['product_cluster'] == 4)]], axis=0)

    file_list = [file_path2, file_path4, file_path6, file_path8]
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
            print('inin2', len(product_temp))

     df_final = df_final[df_final['Description'].isin(product_temp)]


    mae = np.round(mean_absolute_error(df_final['Quantity'],df_final['predict_Quantity']), 2)
    mape = np.round(mean_absolute_percentage_error(df_final['Quantity'], df_final['predict_Quantity']), 2)
    rmse = np.round(np.sqrt(mean_squared_error(df_final['Quantity'], df_final['predict_Quantity'])), 2)