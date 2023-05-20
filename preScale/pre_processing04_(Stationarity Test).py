import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import datetime
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# model packages
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
file_path = './data/df_final.csv'

def get_adfuller(inputSeries):
    result = adfuller(inputSeries, maxlag=None, regression='c', autolag='AIC',store=False, regresults=False)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    if result[1] < 0.05:
        yn = True
    else:
        yn= False
    return yn

if __name__ == "__main__":
    df = pd.read_csv(file_path)
    df = df.set_index(['YYYYMM'])
    df = df[~df['Demand Pattern'].isna()]
    df['predict_Quantity'] = np.nan
    df.sort_values(['Description','YYYYMM'], inplace=True)
    product_list = df['Description'].unique()
    product_tgt_list = []
    print(df.head())
    # 유의수준 0.05 에서 정상성이 있는 상품만 선별
    for p in product_list:
        df_tgt = df[df['Description'] == p]
        if len(df_tgt) >= 10:
            yn1 = get_adfuller(df_tgt['Quantity'])
            if yn1:
                print(p)
                product_tgt_list.append(p)

    df= df[df['Description'].isin(product_tgt_list)]
    print(df)
    print(len(df[df['Demand Pattern'] == 'Erratic']['Description'].unique()))
    print(len(df[df['Demand Pattern'] == 'Intermittent']['Description'].unique()))
    print(len(df[df['Demand Pattern'] == 'Lumpy']['Description'].unique()))
    print(len(df[df['Demand Pattern'] == 'Smooth']['Description'].unique()))

    df = df[['Description','Quantity','0_cluster','1_cluster','2_cluster','3_cluster','4_cluster','Demand Pattern']]
    df.to_csv('./data/df_tgt.csv')

    # product_final = df['Description'].unique()
    #
    # print(df[df['Description']== product_final[1]])
    #
    # plot_df = df[df['Description']== product_final[1]]
    #
    # plt.figure(figsize=(16, 9))
    # plot_df.Quantity.plot(kind='bar')
    # plt.ylabel('Quantity')
    # plt.show()