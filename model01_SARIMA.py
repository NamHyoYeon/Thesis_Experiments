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

file_path = './data/df_final_with_bf1mm.csv'

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    df = pd.read_csv(file_path)

    # 전월 값이 있는 ROW 부터 모델에 사용
    df = df[~df['Quantity_bf1mm'].isna()]
    df = df.set_index(['YYYYMM'])
    df['predict_Quantity'] = np.nan
    print(df.head())
    df.sort_values(['Description','YYYYMM'], inplace=True)

    # 변수 정하기
    df_tgt = df[['Description','Quantity','0_cluster','1_cluster','2_cluster','3_cluster','4_cluster','Erratic','Intermittent','Lumpy','Smooth']]
    print(df_tgt.head())

    # Description(상품) 별로 예측, 가장 마지막 달을 기준으로 train
    product_list = df_tgt['Description'].unique()
    df_tgt['predict_Quantity'] = np.nan

    bf = datetime.datetime.now()

    grouping = df_tgt.groupby('Description').count()
    print(len(grouping[grouping['Quantity'] >=5]))

    # 1. quantity + clustering 값 모두 feature 로 사용
    for product in product_list:
        if len(df_tgt[df_tgt['Description'] == product]) > 5:
            max_index = df_tgt[df_tgt['Description'] == product].index.max()
            second_index = df_tgt[(df_tgt['Description'] == product) & (df_tgt.index != max_index)].index.max()
            train = df_tgt[df_tgt['Description'] == product].loc[:max_index]
            test = df_tgt[df_tgt['Description'] == product].loc[max_index:]
            print(max_index)

            # Fit an SARIMAX model to the training data
            model = SARIMAX(train['Quantity'], exog=train[['0_cluster', '1_cluster', '2_cluster', '3_cluster', '4_cluster']], order=(1, 0, 0))
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
        break

    df_final = df_tgt[~df_tgt['predict_Quantity'].isna()]
    df_final['predict_Quantity'] = df_final['predict_Quantity'].astype('int64')

    df_final.to_csv(r'./data/result/df_final_sarima_1.csv')

    # 2. quantity 만 feature 로 사용
    for product in product_list:
        if len(df_tgt[df_tgt['Description'] == product]) > 5:
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

    df_final.to_csv(r'./data/result/df_final_sarima_2.csv')

    af = datetime.datetime.now()