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

    df_final = df_tgt[~df_tgt['predict_Quantity'].isna()]
    df_final['predict_Quantity'] = df_final['predict_Quantity'].astype('int64')

    # MAE 구하기
    mae1_total = mean_absolute_error(df_final['Quantity'], df_final['predict_Quantity'])
    mae1_Erratic = mean_absolute_error(df_final[df_final['Erratic'] == 1]['Quantity'], df_final[df_final['Erratic'] == 1]['predict_Quantity'])
    mae1_Intermittent = mean_absolute_error(df_final[df_final['Intermittent'] == 1]['Quantity'], df_final[df_final['Intermittent'] == 1]['predict_Quantity'])
    mae1_Lumpy = mean_absolute_error(df_final[df_final['Lumpy'] == 1]['Quantity'], df_final[df_final['Lumpy'] == 1]['predict_Quantity'])
    mae1_Smooth = mean_absolute_error(df_final[df_final['Smooth'] == 1]['Quantity'], df_final[df_final['Smooth'] == 1]['predict_Quantity'])

    # RMSE 구하기
    rmse1_total = np.sqrt(mean_squared_error(df_final['Quantity'], df_final['predict_Quantity']))
    rmse1_Erratic = np.sqrt(mean_squared_error(df_final[df_final['Erratic'] == 1]['Quantity'], df_final[df_final['Erratic'] == 1]['predict_Quantity']))
    rmse1_Intermittent = np.sqrt(mean_squared_error(df_final[df_final['Intermittent'] == 1]['Quantity'], df_final[df_final['Intermittent'] == 1]['predict_Quantity']))
    rmse1_Lumpy = np.sqrt(mean_squared_error(df_final[df_final['Lumpy'] == 1]['Quantity'], df_final[df_final['Lumpy'] == 1]['predict_Quantity']))
    rmse1_Smooth = np.sqrt(mean_squared_error(df_final[df_final['Smooth'] == 1]['Quantity'], df_final[df_final['Smooth'] == 1]['predict_Quantity']))

    # MAPE 구하기
    mape1_total = mean_absolute_percentage_error(df_final['Quantity'], df_final['predict_Quantity'])
    mape1_Erratic = mean_absolute_percentage_error(df_final[df_final['Erratic'] == 1]['Quantity'], df_final[df_final['Erratic'] == 1]['predict_Quantity'])
    mape1_Intermittent = mean_absolute_percentage_error(df_final[df_final['Intermittent'] == 1]['Quantity'], df_final[df_final['Intermittent'] == 1]['predict_Quantity'])
    mape1_Lumpy = mean_absolute_percentage_error(df_final[df_final['Lumpy'] == 1]['Quantity'], df_final[df_final['Lumpy'] == 1]['predict_Quantity'])
    mape1_Smooth = mean_absolute_percentage_error(df_final[df_final['Smooth'] == 1]['Quantity'], df_final[df_final['Smooth'] == 1]['predict_Quantity'])

    # R2 스코어 구하기
    r2s_1_total = r2_score(df_final['Quantity'], df_final['predict_Quantity'])
    r2s_1_Erratic = r2_score(df_final[df_final['Erratic'] == 1]['Quantity'], df_final[df_final['Erratic'] == 1]['predict_Quantity'])
    r2s_1_Intermittent = r2_score(df_final[df_final['Intermittent'] == 1]['Quantity'], df_final[df_final['Intermittent'] == 1]['predict_Quantity'])
    r2s_1_Lumpy = r2_score(df_final[df_final['Lumpy'] == 1]['Quantity'], df_final[df_final['Lumpy'] == 1]['predict_Quantity'])
    r2s_1_Smooth = r2_score(df_final[df_final['Smooth'] == 1]['Quantity'], df_final[df_final['Smooth'] == 1]['predict_Quantity'])

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

    # MAE 구하기
    mae2_total = mean_absolute_error(df_final['Quantity'], df_final['predict_Quantity'])
    mae2_Erratic = mean_absolute_error(df_final[df_final['Erratic'] == 1]['Quantity'], df_final[df_final['Erratic'] == 1]['predict_Quantity'])
    mae2_Intermittent = mean_absolute_error(df_final[df_final['Intermittent'] == 1]['Quantity'], df_final[df_final['Intermittent'] == 1]['predict_Quantity'])
    mae2_Lumpy = mean_absolute_error(df_final[df_final['Lumpy'] == 1]['Quantity'], df_final[df_final['Lumpy'] == 1]['predict_Quantity'])
    mae2_Smooth = mean_absolute_error(df_final[df_final['Smooth'] == 1]['Quantity'], df_final[df_final['Smooth'] == 1]['predict_Quantity'])

    # RMSE 구하기
    rmse2_total = np.sqrt(mean_squared_error(df_final['Quantity'], df_final['predict_Quantity']))
    rmse2_Erratic = np.sqrt(mean_squared_error(df_final[df_final['Erratic'] == 1]['Quantity'], df_final[df_final['Erratic'] == 1]['predict_Quantity']))
    rmse2_Intermittent = np.sqrt(mean_squared_error(df_final[df_final['Intermittent'] == 1]['Quantity'], df_final[df_final['Intermittent'] == 1]['predict_Quantity']))
    rmse2_Lumpy = np.sqrt(mean_squared_error(df_final[df_final['Lumpy'] == 1]['Quantity'], df_final[df_final['Lumpy'] == 1]['predict_Quantity']))
    rmse2_Smooth = np.sqrt(mean_squared_error(df_final[df_final['Smooth'] == 1]['Quantity'], df_final[df_final['Smooth'] == 1]['predict_Quantity']))

    # MAPE 구하기
    mape2_total = mean_absolute_percentage_error(df_final['Quantity'], df_final['predict_Quantity'])
    mape2_Erratic = mean_absolute_percentage_error(df_final[df_final['Erratic'] == 1]['Quantity'], df_final[df_final['Erratic'] == 1]['predict_Quantity'])
    mape2_Intermittent = mean_absolute_percentage_error(df_final[df_final['Intermittent'] == 1]['Quantity'], df_final[df_final['Intermittent'] == 1]['predict_Quantity'])
    mape2_Lumpy = mean_absolute_percentage_error(df_final[df_final['Lumpy'] == 1]['Quantity'], df_final[df_final['Lumpy'] == 1]['predict_Quantity'])
    mape2_Smooth = mean_absolute_percentage_error(df_final[df_final['Smooth'] == 1]['Quantity'], df_final[df_final['Smooth'] == 1]['predict_Quantity'])

    # R2 스코어 구하기
    r2s_2_total = r2_score(df_final['Quantity'], df_final['predict_Quantity'])
    r2s_2_Erratic = r2_score(df_final[df_final['Erratic'] == 1]['Quantity'], df_final[df_final['Erratic'] == 1]['predict_Quantity'])
    r2s_2_Intermittent = r2_score(df_final[df_final['Intermittent'] == 1]['Quantity'], df_final[df_final['Intermittent'] == 1]['predict_Quantity'])
    r2s_2_Lumpy = r2_score(df_final[df_final['Lumpy'] == 1]['Quantity'], df_final[df_final['Lumpy'] == 1]['predict_Quantity'])
    r2s_2_Smooth = r2_score(df_final[df_final['Smooth'] == 1]['Quantity'], df_final[df_final['Smooth'] == 1]['predict_Quantity'])

    df_final.to_csv(r'./data/result/df_final_sarima_2.csv')

    # 예측 건수 확인
    final_product_cnt = len(df_final['Description'].unique())
    final_product_Erratic_cnt = len(df_final[df_final['Erratic'] == 1]['Description'].unique())
    final_product_Intermittent_cnt = len(df_final[df_final['Intermittent'] == 1]['Description'].unique())
    final_product_Lumpy_cnt = len(df_final[df_final['Lumpy'] == 1]['Description'].unique())
    final_product_Smooth_cnt = len(df_final[df_final['Smooth'] == 1]['Description'].unique())

    af = datetime.datetime.now()