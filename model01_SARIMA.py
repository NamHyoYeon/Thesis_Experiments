import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import datetime
import numpy as np

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

    i=0
    for product in product_list:
        i = i+1
        max_index = df_tgt[df_tgt['Description'] == product].index.max()
        train = df_tgt[df_tgt['Description'] == product].loc[:max_index]
        test = df_tgt[df_tgt['Description'] == product].loc[max_index:]
        print(max_index)

        # Fit an ARIMAX model to the training data
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
        ## 왜 업데이트가 안되지??
        df_tgt[df_tgt['Description'] == product].loc[max_index, 'predict_Quantity'] = 0

        if i >= 10:
            print(predict_value)
            print(df_tgt[df_tgt['Description'] == product].loc[max_index, 'Quantity'])
            print(df_tgt[df_tgt['Description'] == product].loc[max_index,'predict_Quantity'])
            break


    print(type(df.iloc[0,0]))
    print(df_tgt[df_tgt['predict_Quantity'].isna()])