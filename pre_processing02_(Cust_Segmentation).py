import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

file_path = './data/Online Retail Dataset.csv'

if __name__ == "__main__":
    df = pd.read_csv(file_path)
    df['UnitPrice'] = df['UnitPrice'].astype('int')
    df['CustomerID'] = df['CustomerID'].astype('str')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['YYYY'] = df['InvoiceDate'].dt.year
    df['MM'] = df['InvoiceDate'].dt.month
    df['DD'] = df['InvoiceDate'].dt.day
    df['YYYYMM'] = df['YYYY'].astype('str') + df['MM'].astype('str').apply(lambda x: str(x).zfill(2))
    df['YYYYMMDD'] = (df['YYYY'].astype('str') + df['MM'].astype('str').apply(lambda x: str(x).zfill(2)) + df['DD'].apply(lambda x: str(x).zfill(2)).astype('str')).astype(int)
    df['Amount'] = df['Quantity'] * df['UnitPrice']

    print(df.info())

    # 1. RFM (Recency, Frequency, Monetary) Calculate
    # 1-1. Calculate Recency (얼마나 최근에 구매 했는가?)
    df_grouping1 = df.groupby('CustomerID')['InvoiceDate'].max().reset_index()
    df_grouping1.columns = ['CustomerID','Max_InvoiceDate']
    print(df_grouping1.head())
    df_grouping1['Recency'] = (df['InvoiceDate'].max() - df_grouping1['Max_InvoiceDate']).dt.days

    # 1-2. Calculate Frequency (얼마나 자주 구매 했는가?)
    df_grouping2 = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
    df_grouping2.columns = ['CustomerID','Frequency']
    print(df_grouping2.head())

    # 1-3. Calculate Monetary (얼마나 많은 금액을 지출 했는가?)
    df_grouping3 = df.groupby('CustomerID')['Amount'].sum().reset_index()
    df_grouping3.columns = ['CustomerID','Monetary']
    print(df_grouping3)

    # 1-4. RFM Merge
    df_grouping = pd.merge(df_grouping1, df_grouping2, how='left', on='CustomerID')
    df_grouping = pd.merge(df_grouping, df_grouping3, how='left', on='CustomerID')
    del df_grouping['Max_InvoiceDate']

    print(df_grouping)

    # 2. RFM + K-Clustering
    # 2-1. Additional Information per CustomerID add
    # First_Purchase
    df_grouping4 = df.groupby('CustomerID')['InvoiceDate'].min().dt.month.reset_index()
    df_grouping4.columns = ['CustomerID','First_Purchase']
    print(df_grouping4.head())

    # Max Amount
    df_grouping5 = df.groupby('CustomerID')['Amount'].max().reset_index()
    df_grouping5.columns = ['CustomerID','MAX_Amount']
    print(df_grouping5.head())

    # Min Amount
    df_grouping6 = df.groupby('CustomerID')['Amount'].min().reset_index()
    df_grouping6.columns = ['CustomerID','MIN_Amount']
    print(df_grouping6.head())

    # Mean Amount
    df_grouping7 = df.groupby('CustomerID')['Amount'].mean().reset_index()
    df_grouping7.columns = ['CustomerID','MEAN_Amount']
    print(df_grouping7.head())

    # 2-2 Merge
    df_grouping = pd.merge(df_grouping,df_grouping4, how='left', on='CustomerID')
    df_grouping = pd.merge(df_grouping,df_grouping5, how='left', on='CustomerID')
    df_grouping = pd.merge(df_grouping,df_grouping6, how='left', on='CustomerID')
    df_grouping = pd.merge(df_grouping,df_grouping7, how='left', on='CustomerID')

    df_grouping.describe()

    # 2-3 K-means Clustering
    # Clustering 5 segments
    print(df_grouping.info())
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(df_grouping[['Recency','Frequency','Monetary','First_Purchase','MAX_Amount','MIN_Amount','MEAN_Amount']])
    df_grouping['cluster'] = kmeans.labels_
    print(df_grouping.head())