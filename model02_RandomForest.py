import pandas as pd
import numpy as np
import datetime

# plot packages
import matplotlib.pyplot as plt

# model packages
from sklearn.ensemble import RandomForestRegressor

# metrics for model

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

file_path = './data/df_final_with_bf1mm.csv'

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

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

    # 1. 전월 quantity + 전월 clustering 값 모두 feature 로 사용
    for p in product_list:
        df_tgt = df[df['Description'] == p]
        max_index = df_tgt.index.max()

        if len(df_tgt) > 5:
            target_variable = 'Quantity'
            features = ['Quantity_bf1mm', '0_cluster_bf1mm', '1_cluster_bf1mm', '2_cluster_bf1mm', '3_cluster_bf1mm', '4_cluster_bf1mm']

            X_train = df_tgt[df_tgt['Description'] == p].loc[:max_index,features]
            X_test = df_tgt[df_tgt['Description'] == p].loc[max_index:, features]
            y_train = df_tgt[df_tgt['Description'] == p].loc[:max_index,target_variable]
            y_test = df_tgt[df_tgt['Description'] == p].loc[max_index:, target_variable]

            rf = RandomForestRegressor(n_estimators=100, max_depth= 5, random_state=42)
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_test)
            print(y_pred)

            df.loc[(df['Description'] == p) & (df.index == max_index), 'predict_Quantity'] = y_pred[0]

    df_val = df[~df['predict_Quantity'].isna()]

    # MAE 구하기
    mae1_total = mean_absolute_error(df_val['Quantity'], df_val['predict_Quantity'])
    mae1_Erratic = mean_absolute_error(df_val[df_val['Erratic'] == 1]['Quantity'], df_val[df_val['Erratic'] == 1]['predict_Quantity'])
    mae1_Intermittent = mean_absolute_error(df_val[df_val['Intermittent'] == 1]['Quantity'], df_val[df_val['Intermittent'] == 1]['predict_Quantity'])
    mae1_Lumpy = mean_absolute_error(df_val[df_val['Lumpy'] == 1]['Quantity'], df_val[df_val['Lumpy'] == 1]['predict_Quantity'])
    mae1_Smooth = mean_absolute_error(df_val[df_val['Smooth'] == 1]['Quantity'], df_val[df_val['Smooth'] == 1]['predict_Quantity'])

    # RMSE 구하기
    rmse1_total = np.sqrt(mean_squared_error(df_val['Quantity'], df_val['predict_Quantity']))
    rmse1_Erratic = np.sqrt(mean_squared_error(df_val[df_val['Erratic'] == 1]['Quantity'], df_val[df_val['Erratic'] == 1]['predict_Quantity']))
    rmse1_Intermittent = np.sqrt(mean_squared_error(df_val[df_val['Intermittent'] == 1]['Quantity'], df_val[df_val['Intermittent'] == 1]['predict_Quantity']))
    rmse1_Lumpy = np.sqrt(mean_squared_error(df_val[df_val['Lumpy'] == 1]['Quantity'], df_val[df_val['Lumpy'] == 1]['predict_Quantity']))
    rmse1_Smooth = np.sqrt(mean_squared_error(df_val[df_val['Smooth'] == 1]['Quantity'], df_val[df_val['Smooth'] == 1]['predict_Quantity']))

    # MAPE 구하기
    mape1_total = mean_absolute_percentage_error(df_val['Quantity'], df_val['predict_Quantity'])
    mape1_Erratic = mean_absolute_percentage_error(df_val[df_val['Erratic'] == 1]['Quantity'], df_val[df_val['Erratic'] == 1]['predict_Quantity'])
    mape1_Intermittent = mean_absolute_percentage_error(df_val[df_val['Intermittent'] == 1]['Quantity'], df_val[df_val['Intermittent'] == 1]['predict_Quantity'])
    mape1_Lumpy = mean_absolute_percentage_error(df_val[df_val['Lumpy'] == 1]['Quantity'], df_val[df_val['Lumpy'] == 1]['predict_Quantity'])
    mape1_Smooth = mean_absolute_percentage_error(df_val[df_val['Smooth'] == 1]['Quantity'], df_val[df_val['Smooth'] == 1]['predict_Quantity'])

    # R2 스코어 구하기
    r2s_1_total = r2_score(df_val['Quantity'], df_val['predict_Quantity'])
    r2s_1_Erratic = r2_score(df_val[df_val['Erratic'] == 1]['Quantity'], df_val[df_val['Erratic'] == 1]['predict_Quantity'])
    r2s_1_Intermittent = r2_score(df_val[df_val['Intermittent'] == 1]['Quantity'], df_val[df_val['Intermittent'] == 1]['predict_Quantity'])
    r2s_1_Lumpy = r2_score(df_val[df_val['Lumpy'] == 1]['Quantity'], df_val[df_val['Lumpy'] == 1]['predict_Quantity'])
    r2s_1_Smooth = r2_score(df_val[df_val['Smooth'] == 1]['Quantity'], df_val[df_val['Smooth'] == 1]['predict_Quantity'])

    df_val.to_csv(r'./data/result/df_final_randomforest_1.csv')

    # 2. 전월 quantity 만 feature 로 사용
    for p in product_list:
        df_tgt = df[df['Description'] == p]
        max_index = df_tgt.index.max()

        if len(df_tgt) > 5:
            target_variable = 'Quantity'
            features = ['Quantity_bf1mm']

            X_train = df_tgt[df_tgt['Description'] == p].loc[:max_index,features]
            X_test = df_tgt[df_tgt['Description'] == p].loc[max_index:, features]
            y_train = df_tgt[df_tgt['Description'] == p].loc[:max_index,target_variable]
            y_test = df_tgt[df_tgt['Description'] == p].loc[max_index:, target_variable]

            rf = RandomForestRegressor(n_estimators=100, max_depth= 5, random_state=42)
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_test)
            print(y_pred)

            df.loc[(df['Description'] == p) & (df.index == max_index), 'predict_Quantity'] = y_pred[0]

    df_val = df[~df['predict_Quantity'].isna()]

    # MAE 구하기
    mae2_total = mean_absolute_error(df_val['Quantity'], df_val['predict_Quantity'])
    mae2_Erratic = mean_absolute_error(df_val[df_val['Erratic'] == 1]['Quantity'], df_val[df_val['Erratic'] == 1]['predict_Quantity'])
    mae2_Intermittent = mean_absolute_error(df_val[df_val['Intermittent'] == 1]['Quantity'], df_val[df_val['Intermittent'] == 1]['predict_Quantity'])
    mae2_Lumpy = mean_absolute_error(df_val[df_val['Lumpy'] == 1]['Quantity'], df_val[df_val['Lumpy'] == 1]['predict_Quantity'])
    mae2_Smooth = mean_absolute_error(df_val[df_val['Smooth'] == 1]['Quantity'], df_val[df_val['Smooth'] == 1]['predict_Quantity'])

    # RMSE 구하기
    rmse2_total = np.sqrt(mean_squared_error(df_val['Quantity'], df_val['predict_Quantity']))
    rmse2_Erratic = np.sqrt(mean_squared_error(df_val[df_val['Erratic'] == 1]['Quantity'], df_val[df_val['Erratic'] == 1]['predict_Quantity']))
    rmse2_Intermittent = np.sqrt(mean_squared_error(df_val[df_val['Intermittent'] == 1]['Quantity'], df_val[df_val['Intermittent'] == 1]['predict_Quantity']))
    rmse2_Lumpy = np.sqrt(mean_squared_error(df_val[df_val['Lumpy'] == 1]['Quantity'], df_val[df_val['Lumpy'] == 1]['predict_Quantity']))
    rmse2_Smooth = np.sqrt(mean_squared_error(df_val[df_val['Smooth'] == 1]['Quantity'], df_val[df_val['Smooth'] == 1]['predict_Quantity']))

    # MAPE 구하기
    mape2_total = mean_absolute_percentage_error(df_val['Quantity'], df_val['predict_Quantity'])
    mape2_Erratic = mean_absolute_percentage_error(df_val[df_val['Erratic'] == 1]['Quantity'], df_val[df_val['Erratic'] == 1]['predict_Quantity'])
    mape2_Intermittent = mean_absolute_percentage_error(df_val[df_val['Intermittent'] == 1]['Quantity'], df_val[df_val['Intermittent'] == 1]['predict_Quantity'])
    mape2_Lumpy = mean_absolute_percentage_error(df_val[df_val['Lumpy'] == 1]['Quantity'], df_val[df_val['Lumpy'] == 1]['predict_Quantity'])
    mape2_Smooth = mean_absolute_percentage_error(df_val[df_val['Smooth'] == 1]['Quantity'], df_val[df_val['Smooth'] == 1]['predict_Quantity'])

    # R2 스코어 구하기
    r2s_2_total = r2_score(df_val['Quantity'], df_val['predict_Quantity'])
    r2s_2_Erratic = r2_score(df_val[df_val['Erratic'] == 1]['Quantity'], df_val[df_val['Erratic'] == 1]['predict_Quantity'])
    r2s_2_Intermittent = r2_score(df_val[df_val['Intermittent'] == 1]['Quantity'], df_val[df_val['Intermittent'] == 1]['predict_Quantity'])
    r2s_2_Lumpy = r2_score(df_val[df_val['Lumpy'] == 1]['Quantity'], df_val[df_val['Lumpy'] == 1]['predict_Quantity'])
    r2s_2_Smooth = r2_score(df_val[df_val['Smooth'] == 1]['Quantity'], df_val[df_val['Smooth'] == 1]['predict_Quantity'])

    df_val.to_csv(r'./data/result/df_final_randomforest_2.csv')

    # 예측 건수 확인
    final_product_cnt = len(df_val['Description'].unique())
    final_product_Erratic_cnt = len(df_val[df_val['Erratic'] == 1]['Description'].unique())
    final_product_Intermittent_cnt = len(df_val[df_val['Intermittent'] == 1]['Description'].unique())
    final_product_Lumpy_cnt = len(df_val[df_val['Lumpy'] == 1]['Description'].unique())
    final_product_Smooth_cnt = len(df_val[df_val['Smooth'] == 1]['Description'].unique())

    af = datetime.datetime.now()

    df_val.to_csv(r'./data/result/df_final_randomforest.csv')