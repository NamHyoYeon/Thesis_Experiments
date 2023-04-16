import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = './data/Online Retail Dataset.csv'

# 월별로 그룹핑 후 ADI, CV 구하기
if __name__ == "__main__":
    # 1-1. (Pre scaling) Data Load
    df = pd.read_csv(file_path)
    df['UnitPrice'] = df['UnitPrice'].astype('int')
    df['CustomerID'] = df['CustomerID'].astype('str')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['YYYY'] = df['InvoiceDate'].dt.year
    df['MM'] = df['InvoiceDate'].dt.month
    df['DD'] = df['InvoiceDate'].dt.day
    df['YYYYMM'] = df['YYYY'].astype('str') +'-' + df['MM'].astype('str').apply(lambda x: str(x).zfill(2))
    print(df.info())

    # 1-2 (Pre scaling) Data Grouping per month
    df_qnt = df.groupby(['Description','YYYYMM'])['Quantity'].sum().reset_index()
    df_cnt = df.groupby(['Description', 'YYYYMM'])['Quantity'].count().reset_index()
    df_cnt.columns = ['Description','YYYYMM','CNT']

    df_yyyymm = pd.merge(df_qnt,df_cnt, on=('Description','YYYYMM'), how='inner')
    df_yyyymm = df_yyyymm.sort_values(by=['Description','YYYYMM'], ascending=[True,True])
    df_yyyymm['YYYYMM_bf'] = df_yyyymm.groupby('Description')['YYYYMM'].shift(1)
    df_yyyymm['diff'] = (pd.to_datetime(df_yyyymm['YYYYMM'])-pd.to_datetime(df_yyyymm['YYYYMM_bf']))
    df_yyyymm['diff'] = (df_yyyymm['diff'].dt.days).div(30.0)

    print(df_yyyymm)

    # 2. ADI Calculate
    df_yyyymm_cnt = df_yyyymm.groupby('Description')['YYYYMM'].count().reset_index()
    df_yyyymm_cnt.columns = ['Description','CNT']
    df_yyyymm_diff_sum = df_yyyymm.groupby('Description')['diff'].sum().reset_index()
    df_yyyymm_diff_sum.columns = ['Description','DIFF_SUM']

    df_adi_cal = pd.merge(df_yyyymm_cnt, df_yyyymm_diff_sum, how='inner')
    df_adi_cal['ADI'] = df_adi_cal['DIFF_SUM'] / df_adi_cal['CNT']

    print(df_adi_cal[df_adi_cal['ADI'] >= 1.32].sort_values(by=['CNT'], ascending=[False]))

    # 3. CV Calculate
    df_yyyymm_std = df_yyyymm.groupby('Description')['Quantity'].std().reset_index()
    df_yyyymm_std.columns = ['Description', 'STD']
    df_yyyymm_mean = df_yyyymm.groupby('Description')['Quantity'].mean().reset_index()
    df_yyyymm_mean.columns = ['Description', 'MEAN']

    df_cv_cal = pd.merge(df_yyyymm_std, df_yyyymm_mean, how='inner')
    df_cv_cal['CV'] = df_cv_cal['STD'] / df_cv_cal['MEAN']

    print(df_cv_cal[(df_cv_cal['CV'] < 0.49) & (df_cv_cal['CV'] > 0.0)])
    
    # 4. ADI, CV Merge
    df_adi_cv = pd.merge(df_adi_cal, df_cv_cal, how='inner')

    df_adi_cv.loc[(df_adi_cv['ADI'] < 1.32) & (df_adi_cv['ADI'] > 0.0 ) & (df_adi_cv['CV'] < 0.49) & (df_adi_cv['CV'] > 0.0),'Demand Pattern'] = 'Smooth'
    df_adi_cv.loc[(df_adi_cv['ADI'] >= 1.32) & (df_adi_cv['CV'] >= 0.49),'Demand Pattern'] = 'Erratic'
    df_adi_cv.loc[(df_adi_cv['ADI'] >= 1.32) & (df_adi_cv['CV'] < 0.49) & (df_adi_cv['CV'] > 0.0),'Demand Pattern'] = 'Intermittent'
    df_adi_cv.loc[(df_adi_cv['ADI'] < 1.32) & (df_adi_cv['ADI'] > 0.0 ) & (df_adi_cv['CV'] >= 0.49),'Demand Pattern'] = 'Lumpy'


    df_adi_cv_result = df_adi_cv[['Description','Demand Pattern', 'CNT']]
    df_adi_cv_result[~df_adi_cv_result['Demand Pattern'].isna()]
    print(df_adi_cv_result.groupby('Demand Pattern')['Description'].count())

    # 일정한 수량의 수요가 규칙적으로 발생
    print(df_adi_cv_result[df_adi_cv_result['Demand Pattern'] == 'Smooth'].sort_values(by='CNT', ascending=False).head())
    df_yyyymm[df_yyyymm['Description'] == 'QUEENS GUARD COFFEE MUG'].plot(x='YYYYMM', y= 'Quantity', kind='bar')

    # 일정하지 않은 수량의 수요가 규칙적으로 발생
    print(df_adi_cv_result[df_adi_cv_result['Demand Pattern'] == 'Erratic'].sort_values(by='CNT', ascending=False).head())
    df_yyyymm[df_yyyymm['Description'] == 'TOAST ITS - HAPPY BIRTHDAY'].plot(x='YYYYMM', y= 'Quantity', kind='bar')

    # 일정한 수량의 수요가 불규칙적으로 발생
    print(df_adi_cv_result[df_adi_cv_result['Demand Pattern'] == 'Intermittent'].sort_values(by='CNT', ascending=False).head())
    df_yyyymm[df_yyyymm['Description'] == 'WHITE STITCHED CUSHION COVER'].plot(x='YYYYMM', y= 'Quantity', kind='bar')

    # 일정하지 않은 수량의 수요가 불규칙적으로 발생
    print(df_adi_cv_result[df_adi_cv_result['Demand Pattern'] == 'Lumpy'].sort_values(by='CNT', ascending=False).head(20))
    df_yyyymm[df_yyyymm['Description'] == 'BLACK/BLUE POLKADOT UMBRELLA'].plot(x='YYYYMM', y= 'Quantity', kind='bar')

    df_final = pd.merge(df_yyyymm, df_adi_cv_result, on='Description', how='left')
    df_final = df_final[['YYYYMM','Description','Quantity','Demand Pattern']]

    print(df_final.info())
    df_final.to_csv(r'./data/df_yyyymm01.csv')
    print("done")