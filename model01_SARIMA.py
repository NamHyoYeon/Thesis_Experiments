import pandas as pd
import numpy as np
import datetime

# plot packages
import matplotlib.pyplot as plt

# model packages
from statsmodels.tsa.statespace.sarimax import SARIMAX

# metrics for model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

file_path = './data/df_final.csv'

if __name__ == "__main__":
    df = pd.read_csv(file_path)
    print(df.info())

    # Demand Pattern 존재 하는 Row 만 필터링
    df = df[~df['Demand Pattern'].isna()]
    # 49,865 건
    print(df.info())

    df['YYYYMM'] = df['YYYYMM'].astype(str)
    df['YYYYMM'] = df['YYYYMM'].str[0:4] + '-' + df['YYYYMM'].str[4:6]
    df['YYYYMM'] = pd.to_datetime(df['YYYYMM'], format='%Y/%m')
    print(df.head())

    # One Hot Encoding
    one_hot_encoded = pd.get_dummies(df['Demand Pattern'])
    print(one_hot_encoded)

    df = pd.concat([df, one_hot_encoded], axis=1)
    print(df.info())

    # 변수 정하기
    df_tgt = df[['YYYYMM','Description','Quantity','0_cluster','1_cluster','2_cluster','3_cluster','4_cluster','Erratic','Intermittent','Lumpy','Smooth']]
    print(df_tgt.head())
    df_tgt = df_tgt.set_index('YYYYMM')

    print(df_tgt)

    # Description(상품) 별로 예측, 가장 마지막 달을 기준으로 train
    product_list = df_tgt['Description'].unique()
    df_tgt['predict_Quantity'] = np.nan

    print(df_tgt.info())
    # 변수가 하나도 없을 때
    for product in product_list:
        max_index = df_tgt[df_tgt['Description'] == product].index.max()
        train = df_tgt[df_tgt['Description'] == product].loc[:max_index]
        test = df_tgt[df_tgt['Description'] == product].loc[max_index:]
        print(max_index)

        # Fit an SARIMAX model to the training data
        model = SARIMAX(train['Quantity'], exog=None, order=(1, 0, 0))
        result = model.fit()

        # Make predictions for the test set
        forecast = result.forecast(steps=len(test), exog=None)

        df = forecast.to_frame()
        df['max_index'] = max_index
        df['product'] = product
        df.columns = ['predict_Quantity','max_index','product']
        predict_value = round(df.iloc[0,0],2)
        df_tgt.loc[(df_tgt['Description'] == product) & (df_tgt.index == max_index), 'predict_Quantity'] = predict_value

    df_final = df_tgt[~df_tgt['predict_Quantity'].isna()]
    df_final['predict_Quantity'] = df_final['predict_Quantity'].astype('int64')

    # RMSE 구하기
    df_final['gap'] = (df_final['Quantity'] - df_final['predict_Quantity'])**2
    rmse1 = np.sqrt(df_final['gap'].mean())
    mae1 = mean_absolute_error(df_final['Quantity'], df_final['predict_Quantity'])

    # 변수를 다 넣었을 때
    for product in product_list:
        max_index = df_tgt[df_tgt['Description'] == product].index.max()
        train = df_tgt[df_tgt['Description'] == product].loc[:max_index]
        test = df_tgt[df_tgt['Description'] == product].loc[max_index:]
        print(max_index)

        # Fit an SARIMAX model to the training data
        model = SARIMAX(train['Quantity'], exog=train[
            ['0_cluster', '1_cluster', '2_cluster', '3_cluster', '4_cluster', 'Erratic', 'Intermittent', 'Lumpy',
             'Smooth']], order=(1, 0, 0))
        result = model.fit()

        # Make predictions for the test set
        forecast = result.forecast(steps=len(test), exog=test[
            ['0_cluster', '1_cluster', '2_cluster', '3_cluster', '4_cluster', 'Erratic', 'Intermittent', 'Lumpy',
             'Smooth']])

        df = forecast.to_frame()
        df['max_index'] = max_index
        df['product'] = product
        df.columns = ['predict_Quantity','max_index','product']
        predict_value = round(df.iloc[0,0],2)
        df_tgt.loc[(df_tgt['Description'] == product) & (df_tgt.index == max_index), 'predict_Quantity'] = predict_value

    df_final = df_tgt[~df_tgt['predict_Quantity'].isna()]
    df_final['predict_Quantity'] = df_final['predict_Quantity'].astype('int64')

    # RMSE 구하기
    df_final['gap'] = (df_final['Quantity'] - df_final['predict_Quantity'])**2
    rmse2 = np.sqrt(df_final['gap'].mean())
    # MAE 구하기
    mae2 = mean_absolute_error(df_final['Quantity'], df_final['predict_Quantity'])

    # 고객 클러스터링 변수만 넣었을 때
    for product in product_list:
        max_index = df_tgt[df_tgt['Description'] == product].index.max()
        train = df_tgt[df_tgt['Description'] == product].loc[:max_index]
        test = df_tgt[df_tgt['Description'] == product].loc[max_index:]
        print(max_index)

        # Fit an SARIMAX model to the training data
        model = SARIMAX(train['Quantity'], exog=train[
            ['0_cluster', '1_cluster', '2_cluster', '3_cluster', '4_cluster']], order=(1, 0, 0))
        result = model.fit()

        # Make predictions for the test set
        forecast = result.forecast(steps=len(test), exog=test[
            ['0_cluster', '1_cluster', '2_cluster', '3_cluster', '4_cluster']])

        df = forecast.to_frame()
        df['max_index'] = max_index
        df['product'] = product
        df.columns = ['predict_Quantity','max_index','product']
        predict_value = round(df.iloc[0,0],2)
        df_tgt.loc[(df_tgt['Description'] == product) & (df_tgt.index == max_index), 'predict_Quantity'] = predict_value

    df_final = df_tgt[~df_tgt['predict_Quantity'].isna()]
    df_final['predict_Quantity'] = df_final['predict_Quantity'].astype('int64')

    # RMSE 구하기
    df_final['gap'] = (df_final['Quantity'] - df_final['predict_Quantity'])**2
    rmse3 = np.sqrt(df_final['gap'].mean())
    # MAE 구하기
    mae3 = mean_absolute_error(df_final['Quantity'], df_final['predict_Quantity'])

    # 상품 수요 패턴 변수만 넣었을 때
    for product in product_list:
        max_index = df_tgt[df_tgt['Description'] == product].index.max()
        train = df_tgt[df_tgt['Description'] == product].loc[:max_index]
        test = df_tgt[df_tgt['Description'] == product].loc[max_index:]
        print(max_index)

        # Fit an SARIMAX model to the training data
        model = SARIMAX(train['Quantity'], exog=train[
            ['Erratic', 'Intermittent', 'Lumpy',
             'Smooth']], order=(1, 0, 0))
        result = model.fit()

        # Make predictions for the test set
        forecast = result.forecast(steps=len(test), exog=test[
            ['Erratic', 'Intermittent', 'Lumpy',
             'Smooth']])

        df = forecast.to_frame()
        df['max_index'] = max_index
        df['product'] = product
        df.columns = ['predict_Quantity', 'max_index', 'product']
        predict_value = round(df.iloc[0, 0], 2)
        df_tgt.loc[(df_tgt['Description'] == product) & (df_tgt.index == max_index), 'predict_Quantity'] = predict_value

    df_final = df_tgt[~df_tgt['predict_Quantity'].isna()]
    df_final['predict_Quantity'] = df_final['predict_Quantity'].astype('int64')

    # RMSE 구하기
    df_final['gap'] = (df_final['Quantity'] - df_final['predict_Quantity']) ** 2
    rmse4 = np.sqrt(df_final['gap'].mean())
    # MAE 구하기
    mae4 = mean_absolute_error(df_final['Quantity'], df_final['predict_Quantity'])