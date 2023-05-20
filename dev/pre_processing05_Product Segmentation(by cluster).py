import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import datetime

from sklearn.cluster import KMeans

# model packages
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
file_path = './data/df_final_with_bf1mm_2.csv'

if __name__ == "__main__":
    df = pd.read_csv(file_path)
    group_cluster = df.groupby('Description')[['Quantity','0_cluster','1_cluster','2_cluster','3_cluster','4_cluster']].sum().reset_index()
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(group_cluster[['Quantity','0_cluster','1_cluster','2_cluster','3_cluster','4_cluster']])
    group_cluster['product_cluster'] = kmeans.labels_
    print(group_cluster)

    group_cluster = group_cluster[['Description','product_cluster']]
    print(group_cluster)

    group_cluster.to_csv(r'./data/df_product_cluster.csv')






