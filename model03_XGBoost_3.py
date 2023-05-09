import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import datetime

# model packages
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
file_path = './data/df_final.csv'

if __name__ == "__main__":
    df = pd.read_csv(file_path)
    df = df.set_index(['YYYYMM'])
    df['predict_Quantity'] = np.nan
    df.sort_values(['Description','YYYYMM'], inplace=True)
    product_list = df['Description'].unique()

    bf = datetime.datetime.now()

    # Grid Search
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01],
        'n_estimators': [50, 100, 200]
    }

    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    df_predict = pd.DataFrame()

    bf = datetime.datetime.now()
    for p in product_list:
        print(p)
        df_tgt = df[df['Description'] == p]
        if len(df_tgt) > 5:
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
            not_features = ['Description','0_cluster','1_cluster','2_cluster','3_cluster','4_cluster']
            features = [x for x in df_input.columns if (x != target_variable) & (x not in not_features)]

            x_train = df_input.loc[:index_list[-2],features]
            y_train = df_input.loc[:index_list[-2],target_variable]

            x_test = df_input.loc[index_list[-1]:,features]
            y_test = df_input.loc[index_list[-1]:,target_variable]

            xgb.fit(x_train, y_train)
            df_predict_temp = df_input.loc[index_list[-1]:,:]
            df_predict = pd.concat([df_predict, df_predict_temp], axis=0)
            # df.loc[(df['Description'] == p) & (df.index == max_index), 'predict_Quantity'] = int(np.ceil(y_pred[0]))

    af = datetime.datetime.now()

    df_predict = df_predict.reset_index()
    df_predict['YYYYMM']  = df_predict['index']
    del df_predict['index']
    print(df_predict.info())

    for i in range(len(df_predict)):
        product = df_predict.loc[i,'Description']
        print(product)
        x_predict = df_predict.loc[i:i+1,features]
        y_predict = df_predict.loc[i,target_variable]
        print(y_predict)

        y_pred = xgb.predict(x_predict)
        print(y_pred)
        print(y_predict)

        break


