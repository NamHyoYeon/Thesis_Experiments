import pandas as pd
import numpy as np
import datetime

# plot packages
import matplotlib.pyplot as plt

# model packages
from xgboost import XGBRegressor

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

            xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            xgb.fit(X_train, y_train)

            y_pred = xgb.predict(X_test)
            print(y_pred)

            df.loc[(df['Description'] == p) & (df.index == max_index), 'predict_Quantity'] = y_pred[0]

    df_val = df[~df['predict_Quantity'].isna()]
    # MAE 구하기
    mae1 = mean_absolute_error(df_val['Quantity'], df_val['predict_Quantity'])

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

            xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            xgb.fit(X_train, y_train)

            y_pred = xgb.predict(X_test)
            print(y_pred)

            df.loc[(df['Description'] == p) & (df.index == max_index), 'predict_Quantity'] = y_pred[0]

    df_val = df[~df['predict_Quantity'].isna()]
    # MAE 구하기
    mae2 = mean_absolute_error(df_val['Quantity'], df_val['predict_Quantity'])

    for p in product_list:
        df_tgt = df[df['Description'] == p]
        max_index = df_tgt.index.max()

        if len(df_tgt) > 5:
            target_variable = 'Quantity'
            features = ['Quantity_bf1mm','Erratic', 'Intermittent', 'Lumpy', 'Smooth']

            X_train = df_tgt[df_tgt['Description'] == p].loc[:max_index,features]
            X_test = df_tgt[df_tgt['Description'] == p].loc[max_index:, features]
            y_train = df_tgt[df_tgt['Description'] == p].loc[:max_index,target_variable]
            y_test = df_tgt[df_tgt['Description'] == p].loc[max_index:, target_variable]

            xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            xgb.fit(X_train, y_train)

            y_pred = xgb.predict(X_test)
            print(y_pred)

            df.loc[(df['Description'] == p) & (df.index == max_index), 'predict_Quantity'] = y_pred[0]

    df_val = df[~df['predict_Quantity'].isna()]
    # MAE 구하기
    mae3 = mean_absolute_error(df_val['Quantity'], df_val['predict_Quantity'])

    for p in product_list:
        df_tgt = df[df['Description'] == p]
        max_index = df_tgt.index.max()

        if len(df_tgt) > 5:
            target_variable = 'Quantity'
            features = ['Quantity_bf1mm', '0_cluster_bf1mm', '1_cluster_bf1mm', '2_cluster_bf1mm', '3_cluster_bf1mm', '4_cluster_bf1mm','Erratic', 'Intermittent', 'Lumpy', 'Smooth']

            X_train = df_tgt[df_tgt['Description'] == p].loc[:max_index,features]
            X_test = df_tgt[df_tgt['Description'] == p].loc[max_index:, features]
            y_train = df_tgt[df_tgt['Description'] == p].loc[:max_index,target_variable]
            y_test = df_tgt[df_tgt['Description'] == p].loc[max_index:, target_variable]

            xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            xgb.fit(X_train, y_train)

            y_pred = xgb.predict(X_test)
            print(y_pred)

            df.loc[(df['Description'] == p) & (df.index == max_index), 'predict_Quantity'] = y_pred[0]

    df_val = df[~df['predict_Quantity'].isna()]
    # MAE 구하기
    mae4 = mean_absolute_error(df_val['Quantity'], df_val['predict_Quantity'])

    af = datetime.datetime.now

    print(df_val.head())