import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import datetime

import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima

file_path = './data/df_final.csv'

if __name__ == "__main__":
    df = pd.read_csv(file_path)
    print(df.info())

    # Demand Pattern 존재 하는 Row 만 필터링
    df = df[~df['Demand Pattern'].isna()]
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

    # 상품별 건수 확인
    df_grouping= df_tgt.groupby('Description')['Quantity'].count()
    print(df_grouping.info())

    # 변수 정하기
    df_tgt = df[['YYYYMM','Description','Quantity','0_cluster','1_cluster','2_cluster','3_cluster','4_cluster','Erratic','Intermittent','Lumpy','Smooth']]
    print(df_tgt.head())
    df_tgt = df_tgt.set_index('YYYYMM')

    print(df_tgt)

    # Description 별로 예측
    train = df_tgt[df_tgt['Description'] == '10 COLOUR SPACEBOY PEN'].loc[:'2011-12-01']
    test = df_tgt[df_tgt['Description'] == '10 COLOUR SPACEBOY PEN'].loc['2011-12-01':]

    # Fit an ARIMAX model to the training data
    model = SARIMAX(train['Quantity'], exog=train[['0_cluster','1_cluster','2_cluster','3_cluster','4_cluster','Erratic','Intermittent','Lumpy','Smooth']], order=(1, 0, 0))
    result = model.fit()

    # Make predictions for the test set
    forecast = result.forecast(steps=len(test), exog=test[['0_cluster','1_cluster','2_cluster','3_cluster','4_cluster','Erratic','Intermittent','Lumpy','Smooth']])

    # Print the forecasted values
    print(forecast)

    test

    # Fit an ARIMAX model to the training data
    model = SARIMAX(train['Quantity'], exog=train[['0_cluster','1_cluster','2_cluster','3_cluster','4_cluster']], order=(1, 0, 0))
    result = model.fit()

    # Make predictions for the test set
    forecast = result.forecast(steps=len(test), exog=test[['0_cluster','1_cluster','2_cluster','3_cluster','4_cluster']])

    # Print the forecasted values
    print(forecast)