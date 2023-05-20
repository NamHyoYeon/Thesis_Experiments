import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import datetime

from sklearn.cluster import KMeans
file_path = './data/df_tgt.csv'

if __name__ == "__main__":
    df = pd.read_csv(file_path)
    pattern = df['Demand Pattern'].unique()
    df_final = pd.DataFrame()

    for p in pattern:
            if p != 'Intermittent':
                df_tgt = df[df['Demand Pattern'] == p]
                group_cluster = df_tgt.groupby('Description')[
                    ['Quantity', '0_cluster', '1_cluster', '2_cluster', '3_cluster', '4_cluster']].sum().reset_index()
                kmeans = KMeans(n_clusters=5)
                kmeans.fit(group_cluster[['Quantity', '0_cluster', '1_cluster', '2_cluster', '3_cluster', '4_cluster']])
                group_cluster['product_cluster'] = kmeans.labels_
                group_cluster = group_cluster[['Description', 'product_cluster']]
                df_tgt = pd.merge(df_tgt, group_cluster, how='left', on='Description')
            elif p == 'Intermittent':
                print("inin")
                df_tgt = df[df['Demand Pattern'] == p]
                group_cluster = df_tgt.groupby('Description')[
                    ['Quantity', '0_cluster', '1_cluster', '2_cluster', '3_cluster', '4_cluster']].sum().reset_index()
                group_cluster['product_cluster'] = 0
                group_cluster = group_cluster[['Description', 'product_cluster']]
                df_tgt = pd.merge(df_tgt, group_cluster, how='left', on='Description')

            df_final = pd.concat([df_final,df_tgt], axis=0)

    print(df_final.head())
    df_final[df_final['Demand Pattern'] == 'Intermittent']['product_cluster'] = 0
    print(df_final[df_final['Demand Pattern'] == 'Intermittent'])

    df_final.to_csv(r'./data/df_tgt.csv')






