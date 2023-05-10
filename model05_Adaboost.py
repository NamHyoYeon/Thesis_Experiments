import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import datetime

# model packages
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostRegressor

file_path = './data/df_final_with_bf1mm.csv'

if __name__ == "__main__":
    df = pd.read_csv(file_path)
    df = df.set_index(['YYYYMM'])
    df['predict_Quantity'] = np.nan
    df.sort_values(['Description','YYYYMM'], inplace=True)
    product_list = df['Description'].unique()

    bf = datetime.datetime.now()

    for p in product_list:
        print(p)
        df_tgt = df[df['Description'] == p]
        if len(df_tgt) >= 10:
            df_tgt = df_tgt[['Description','Quantity','0_cluster','1_cluster','2_cluster','3_cluster','4_cluster']]
            index_list = df_tgt.index.to_list()
            max_index = index_list[-1]

            df_input = pd.DataFrame()
            look_back = 3

            for i in range(len(index_list)):
                bf_index = 1
                df_temp = df_tgt[(df_tgt.index == index_list[i]) & (df_tgt['Description'] == p)]
                if i >= look_back:
                 for j in range(i-look_back,i):
                    temp = df_tgt[(df_tgt.index == index_list[j]) & (df_tgt['Description'] == p)]
                    temp = temp.reset_index()
                    temp.index = [index_list[i]]
                    temp = temp[['Quantity','0_cluster','1_cluster','2_cluster','3_cluster','4_cluster']]
                    temp = temp.rename(columns=lambda x: f'bf_{bf_index}_{x}')

                    df_temp = pd.concat([df_temp, temp], axis=1)
                    if j == i-1:
                        df_input = pd.concat([df_input, df_temp], axis=0)
                    bf_index = bf_index + 1

            target_variable = 'Quantity'
            # 당월 값은 및 상품명은 제외
            not_features = ['Description','0_cluster','1_cluster','2_cluster','3_cluster','4_cluster']
            tgt_scale = [x for x in df_input.columns if x not in not_features]
            features = [x for x in df_input.columns if (x != target_variable) & (x not in not_features)]

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data_x = scaler.fit_transform(df_input[features])

            x_train = scaled_data_x[:-1,1:]
            x_test = scaled_data_x[-1:,1:]

            y_train = df_input.loc[:index_list[-2], target_variable]
            y_test = df_input.loc[index_list[-1]:, target_variable]

            # create a DecisionTreeRegressor base estimator
            base_estimator = DecisionTreeRegressor(max_depth=4)
            xgb = AdaBoostRegressor(base_estimator=base_estimator, loss='exponential', n_estimators=100, learning_rate=0.01, random_state=42)
            xgb.fit(x_train, y_train)
            y_pred = xgb.predict(x_test)

            print(y_pred)
            print(y_test)

            df.loc[(df['Description'] == p) & (df.index == max_index), 'predict_Quantity'] = int(np.ceil(y_pred[0]))

    af = datetime.datetime.now()
    df_val = df[~df['predict_Quantity'].isna()]
    print(df_val)

    df_val.to_csv(r'./data/result/df_final_adaboost_1.csv')

    for p in product_list:
        print(p)
        df_tgt = df[df['Description'] == p]
        if len(df_tgt) >= 10:
            df_tgt = df_tgt[
                ['Description', 'Quantity']]
            index_list = df_tgt.index.to_list()
            max_index = index_list[-1]

            df_input = pd.DataFrame()
            look_back = 3

            for i in range(len(index_list)):
                bf_index = 1
                df_temp = df_tgt[(df_tgt.index == index_list[i]) & (df_tgt['Description'] == p)]
                if i >= look_back:
                    for j in range(i - look_back, i):
                        temp = df_tgt[(df_tgt.index == index_list[j]) & (df_tgt['Description'] == p)]
                        temp = temp.reset_index()
                        temp.index = [index_list[i]]
                        temp = temp[['Quantity', '0_cluster', '1_cluster', '2_cluster', '3_cluster', '4_cluster']]
                        temp = temp.rename(columns=lambda x: f'bf_{bf_index}_{x}')

                        df_temp = pd.concat([df_temp, temp], axis=1)
                        if j == i - 1:
                            df_input = pd.concat([df_input, df_temp], axis=0)
                        bf_index = bf_index + 1

            target_variable = 'Quantity'
            # 당월 값은 및 상품명은 제외
            not_features = ['Description']
            tgt_scale = [x for x in df_input.columns if x not in not_features]
            features = [x for x in df_input.columns if (x != target_variable) & (x not in not_features)]

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data_x = scaler.fit_transform(df_input[features])

            x_train = scaled_data_x[:-1, 1:]
            x_test = scaled_data_x[-1:, 1:]

            y_train = df_input.loc[:index_list[-2], target_variable]
            y_test = df_input.loc[index_list[-1]:, target_variable]

            # create a DecisionTreeRegressor base estimator
            base_estimator = DecisionTreeRegressor(max_depth=4)
            xgb = AdaBoostRegressor(base_estimator=base_estimator, loss='exponential', n_estimators=100,
                                    learning_rate=0.01, random_state=42)
            xgb.fit(x_train, y_train)
            y_pred = xgb.predict(x_test)

            print(y_pred)
            print(y_test)

            df.loc[(df['Description'] == p) & (df.index == max_index), 'predict_Quantity'] = int(np.ceil(y_pred[0]))

    af = datetime.datetime.now()
    df_val = df[~df['predict_Quantity'].isna()]
    print(df_val)

    df_val.to_csv(r'./data/result/df_final_adaboost_2.csv')


