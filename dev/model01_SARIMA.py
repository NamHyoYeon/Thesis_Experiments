import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import datetime

# model packages
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

file_path = './data/df_final_with_bf1mm.csv'

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

if __name__ == "__main__":
    df = pd.read_csv(file_path)
    df = df.set_index(['YYYYMM'])
    df['predict_Quantity'] = np.nan
    df.sort_values(['Description','YYYYMM'], inplace=True)
    product_list = df['Description'].unique()

    bf = datetime.datetime.now()

    # 1. quantity + clustering 값 모두 feature 로 사용
    for product in product_list:
        df_tgt = df[df['Description'] == product]
        print(df_tgt)

        if len(df_tgt) >= 10:
            max_index = df_tgt.index.max()
            second_index = df_tgt[df_tgt.index != max_index].index.max()

            target_variable = 'Quantity'
            features = ['0_cluster', '1_cluster', '2_cluster', '3_cluster', '4_cluster']
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data_x = scaler.fit_transform(df_tgt[features])

            x_train = scaled_data_x[:-1,1:]
            x_test = scaled_data_x[-1:,1:]

            y_train = df_tgt.loc[:second_index,target_variable]
            y_test = df_tgt.loc[max_index:,target_variable]

            # Fit an SARIMAX model to the training data
            model = SARIMAX(y_train, exog=x_train, order=(1, 0, 0))
            result = model.fit()

            # Make predictions for the test set
            forecast = result.forecast(steps=len(y_test), exorg=x_test)

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
        if len(df_tgt[df_tgt['Description'] == product]) >= 10:
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