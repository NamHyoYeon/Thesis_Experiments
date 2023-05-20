import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import datetime

# model packages
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

file_path = './data/df_tgt_3.csv'

if __name__ == "__main__":
    df = pd.read_csv(file_path)
    df = df.set_index(['YYYYMM'])
    df['predict_Quantity'] = np.nan
    df.sort_values(['Description','YYYYMM'], inplace=True)
    product_list = df['Description'].unique()

    print(df)

    bf = datetime.datetime.now()
    look_back = 5

    for p in product_list:
        print(p)
        df_tgt = df[df['Description'] == p]
        demand_pattern = df_tgt['Demand Pattern'].unique()[0]
        print(demand_pattern)
        df_tgt = df_tgt[
            ['Description', 'Quantity']]
        # 마지막에서 3번째 까지
        index_list = df_tgt.index.to_list()[:-2]
        max_index = index_list[-1]

        df_input = pd.DataFrame()
        for i in range(len(index_list)):
            bf_index = 1
            df_temp = df_tgt[(df_tgt.index == index_list[i]) & (df_tgt['Description'] == p)]
            if i >= look_back:
                for j in range(i - look_back, i):
                    temp = df_tgt[(df_tgt.index == index_list[j]) & (df_tgt['Description'] == p)]
                    temp = temp.reset_index()
                    temp.index = [index_list[i]]
                    temp = temp[['Quantity']]
                    temp = temp.rename(columns=lambda x: f'bf_{bf_index}_{x}')

                    df_temp = pd.concat([df_temp, temp], axis=1)
                    if j == i - 1:
                        df_input = pd.concat([df_input, df_temp], axis=0)
                    bf_index = bf_index + 1

        target_variable = 'Quantity'
        # 당월 값은 및 상품명은 제외
        not_features = ['Description']
        features = [x for x in df_input.columns if (x != target_variable) & (x not in not_features)]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data_x = scaler.fit_transform(df_input[features])

        x_train = scaled_data_x[:-1,1:]
        x_test = scaled_data_x[-1:,1:]

        y_train = df_input.loc[:index_list[-2], target_variable]
        y_test = df_input.loc[index_list[-1]:, target_variable]

        if demand_pattern == 'Smooth':
            lr = 0.05
        elif  demand_pattern == 'Intermittent':
            lr = 0.05
        else :
            lr = 0.02

        xgb = XGBRegressor(objective='reg:squarederror', n_estimators=5, learning_rate=lr, random_state=42, max_depth=3)
        xgb.fit(x_train, y_train)
        y_pred = xgb.predict(x_test)
        print(y_pred)
        print(y_test)

        df.loc[(df['Description'] == p) & (df.index == max_index), 'predict_Quantity'] = int(np.ceil(y_pred[0]))

    af = datetime.datetime.now()
    df_val = df[~df['predict_Quantity'].isna()]
    print(df_val.info())

    df_val = df_val[df_val['Quantity'] >= 1]
    df_val = df_val[df_val['predict_Quantity'] >= 0]

    #df_val.to_csv('./data/result/xgboost3_lookback_2.csv')
    #df_val.to_csv('./data/result/xgboost3_lookback_3.csv')
    #df_val.to_csv('./data/result/xgboost3_lookback_4.csv')
    df_val.to_csv('./data/result/xgboost3_lookback_5.csv')



