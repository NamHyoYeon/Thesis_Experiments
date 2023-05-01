import pandas as pd
import numpy as np
import datetime

file_path = './data/df_final.csv'

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)

    df = pd.read_csv(file_path)
    print(df.info())

    # Demand Pattern 존재 하는 Row 만 필터링
    df = df[~df['Demand Pattern'].isna()]
    # 49,865 건
    print(df.info())

    df['YYYYMM'] = df['YYYYMM'].astype(str)
    df['YYYYMM'] = df['YYYYMM'].str[0:4] + '-' + df['YYYYMM'].str[4:6]
    df['YYYYMM'] = pd.to_datetime(df['YYYYMM'], format='%Y/%m')
    print(df.head())

    # One Hot Encoding
    one_hot_encoded = pd.get_dummies(df['Demand Pattern'])
    print(one_hot_encoded)

    df = pd.concat([df, one_hot_encoded], axis=1)
    print(df.info())

    # 변수 정하기
    df_tgt = df[['YYYYMM', 'Description', 'Quantity', '0_cluster', '1_cluster', '2_cluster', '3_cluster', '4_cluster',
                 'Erratic', 'Intermittent', 'Lumpy', 'Smooth']]
    df_tgt = df_tgt.set_index('YYYYMM')
    df_tgt.sort_values(['Description','YYYYMM'], inplace=True)
    print(df_tgt.head(20))

    product_list = df_tgt['Description'].unique()
    # 3,742 건
    print(product_list)
    print(len(product_list))

    df_tgt[['Quantity_bf1mm','0_cluster_bf1mm','1_cluster_bf1mm','2_cluster_bf1mm','3_cluster_bf1mm','4_cluster_bf1mm']] = np.nan
    print(df_tgt)

    # 상품별 전월 값 가져오기!!
    for p in product_list:
        shifted = df_tgt[df_tgt['Description'] == p].groupby(['Description']).shift(1)
        shifted = shifted[['Quantity', '0_cluster', '1_cluster', '2_cluster', '3_cluster', '4_cluster']]
        shifted['Description'] = p
        shifted.columns = ['Quantity_bf1mm', '0_cluster_bf1mm', '1_cluster_bf1mm', '2_cluster_bf1mm', '3_cluster_bf1mm',
                           '4_cluster_bf1mm', 'Description']

        yyyymm_list = shifted.index.to_list()
        for yyyymm in yyyymm_list:
            yyyymm = yyyymm.strftime("%Y-%m-%d")
            print(yyyymm)
            print(type(yyyymm))

            df_tgt.loc[(df_tgt['Description'] == p) & (df_tgt.index == yyyymm), 'Quantity_bf1mm'] = shifted.loc[yyyymm, 'Quantity_bf1mm']
            df_tgt.loc[(df_tgt['Description'] == p) & (df_tgt.index == yyyymm), '0_cluster_bf1mm'] = shifted.loc[yyyymm, '0_cluster_bf1mm']
            df_tgt.loc[(df_tgt['Description'] == p) & (df_tgt.index == yyyymm), '1_cluster_bf1mm'] = shifted.loc[yyyymm, '1_cluster_bf1mm']
            df_tgt.loc[(df_tgt['Description'] == p) & (df_tgt.index == yyyymm), '2_cluster_bf1mm'] = shifted.loc[yyyymm, '2_cluster_bf1mm']
            df_tgt.loc[(df_tgt['Description'] == p) & (df_tgt.index == yyyymm), '3_cluster_bf1mm'] = shifted.loc[yyyymm, '3_cluster_bf1mm']
            df_tgt.loc[(df_tgt['Description'] == p) & (df_tgt.index == yyyymm), '4_cluster_bf1mm'] = shifted.loc[yyyymm, '4_cluster_bf1mm']

    print(df_tgt.head(10))
    df_tgt.to_csv('./data/df_final_with_bf1mm.csv')



