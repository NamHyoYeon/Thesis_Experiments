import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import datetime

from sklearn.cluster import KMeans
file_path = './data/df_tgt_temp.csv'

def look_back_df(look_back_product, look_df, look_back):
    tgt_df = pd.DataFrame()

    for p in look_back_product:
        print(p)
        look_df_tgt = look_df[look_df['Description'] == p]
        print(look_df_tgt.head())
        look_df_tgt = look_df_tgt[
            ['Description', 'Quantity']]
        index_list = look_df_tgt.index.to_list()
        max_index = index_list[-1]

        print(index_list)

        df_input = pd.DataFrame()
        for i in range(len(index_list)):
            bf_index = 1
            df_temp = look_df_tgt[(look_df_tgt.index == index_list[i]) & (look_df_tgt['Description'] == p)]
            if i >= look_back:
                for j in range(i - look_back, i):
                    temp = look_df_tgt[(look_df_tgt.index == index_list[j]) & (look_df_tgt['Description'] == p)]
                    temp = temp.reset_index()
                    temp.index = [index_list[i]]
                    temp = temp[['Quantity']]
                    temp = temp.rename(columns=lambda x: f'bf_{bf_index}_{x}')

                    df_temp = pd.concat([df_temp, temp], axis=1)
                    if j == i - 1:
                        df_input = pd.concat([df_input, df_temp], axis=0)
                    bf_index = bf_index + 1
        print(df_input)
        print(max_index)
        tgt_df = pd.concat([tgt_df, df_input[df_input.index == max_index]], axis=0)
    return tgt_df

if __name__ == "__main__":
    df = pd.read_csv(file_path)
    pattern = df['Demand Pattern'].unique()
    df_final = pd.DataFrame()

    for p in pattern:
            if p != 'Intermittent':
                df_tgt = df[df['Demand Pattern'] == p]

                group_cluster = df_tgt.groupby(['YYYYMM','Description'])['Quantity'].sum().reset_index()
                group_cluster.set_index('YYYYMM', inplace=True)
                product_list_for = group_cluster['Description'].unique()
                cluster_df = look_back_df(product_list_for,group_cluster,2)

                kmeans = KMeans(n_clusters=5)
                kmeans.fit(cluster_df[['bf_1_Quantity','bf_2_Quantity']])
                cluster_df['product_cluster'] = kmeans.labels_
                cluster_df = cluster_df[['Description', 'product_cluster']]
                df_tgt = pd.merge(df_tgt, cluster_df, how='left', on='Description')

            df_final = pd.concat([df_final,df_tgt], axis=0)

    print(len(df_final[df_final['Demand Pattern'] == 'Intermittent']))

    df_final_temp = df[df['Demand Pattern'] == 'Intermittent']
    df_final_temp['product_cluster']  = 0
    df_final_temp = df_final_temp[['YYYYMM','Description','Quantity','Demand Pattern','product_cluster']]

    df_final= df_final[['YYYYMM','Description','Quantity','Demand Pattern','product_cluster']]
    df_final.groupy('Description','product_cluster').count()

    df_no_duplicates = df_final.drop_duplicates('Description','product_cluster')

    print(df_final.info())

    df_final = pd.concat([df_final,df_final_temp], axis=0)
    df_final_temp = df_final.groupby(['Description','product_cluster']).count().reset_index()
    df_final_temp = df_final_temp[['Description','product_cluster']]

    df = pd.merge(df, df_final_temp, on='Description',how='left')

    df_final = df
    print(df_final)

    df_final.to_csv(r'./data/df_tgt.csv')





