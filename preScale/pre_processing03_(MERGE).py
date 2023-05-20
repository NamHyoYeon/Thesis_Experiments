import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import os

file_path1 = './data/Online Retail Dataset.csv'
file_path2 = './data/df_yyyymm01.csv'
file_path3 = './data/df_grouping_customer.csv'

if __name__ == "__main__":
    df = pd.read_csv(file_path1)
    df['UnitPrice'] = df['UnitPrice'].astype('int')
    df['CustomerID'] = df['CustomerID'].astype('str')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['YYYY'] = df['InvoiceDate'].dt.year
    df['MM'] = df['InvoiceDate'].dt.month
    df['DD'] = df['InvoiceDate'].dt.day
    df['YYYYMM'] = df['YYYY'].astype('str') + df['MM'].astype('str').apply(lambda x: str(x).zfill(2))
    df['YYYYMMDD'] = (df['YYYY'].astype('str') + df['MM'].astype('str').apply(lambda x: str(x).zfill(2)) + df['DD'].apply(lambda x: str(x).zfill(2)).astype('str')).astype(int)
    df['Amount'] = df['Quantity'] * df['UnitPrice']

    # 542,014
    print(df.info())
    df_yyyymm = pd.read_csv(file_path2)
    del df_yyyymm['Unnamed: 0']
    print(df_yyyymm['Description'].nunique())
    print(df_yyyymm.info())

    df_yyyymm['YYYYMM'] = df_yyyymm['YYYYMM'].str.replace('-','')
    print(df_yyyymm.head())

    df_grouping_customer = pd.read_csv(file_path3)
    df_grouping_customer['CustomerID'] = df_grouping_customer['CustomerID'].astype(str)
    del df_grouping_customer['Unnamed: 0']
    print(df_grouping_customer.info())

    df = pd.merge(df, df_grouping_customer, on='CustomerID', how='left')
    df = df[['YYYYMM','Description','cluster','Quantity']]
    grouping = df.groupby(['YYYYMM','Description','cluster'])['Quantity'].sum().reset_index()
    pivot_df = grouping.pivot(columns=('cluster'), values='Quantity')
    pivot_df.fillna(0, inplace=True)
    pivot_df = pivot_df.astype(int)
    pivot_df.columns = ['0_cluster','1_cluster','2_cluster','3_cluster','4_cluster']
    df_grouping_customer = pd.concat([grouping,pivot_df],axis=1)
    del df_grouping_customer['cluster']

    df_grouping_yyyymm = df_grouping_customer.groupby(['Description','YYYYMM'])[['0_cluster','1_cluster','2_cluster','3_cluster','4_cluster']].sum().reset_index()
    print(df_grouping_yyyymm.info())

    # Merge
    df_yyyymm_final = pd.merge(df_yyyymm, df_grouping_yyyymm, on=['Description','YYYYMM'], how='inner')
    print(df_yyyymm_final.info())

    df_yyyymm_final.sort_values(by=['Description','YYYYMM'], inplace=True)

    print(df_yyyymm_final)

    df_yyyymm_final.to_csv('./data/df_final.csv', index=False)
    print("finish")