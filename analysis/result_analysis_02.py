import pandas as pd

pd.set_option('display.max_columns', None)
import numpy as np
import datetime

# plot packages
import matplotlib.pyplot as plt

# metrics for model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error

file_path1 = './data/result/xgboost_lookback_2.csv'
file_path2 = './data/result/xgboost_lookback_3.csv'
file_path3 = './data/result/xgboost_lookback_4.csv'
file_path4 = './data/result/xgboost_lookback_5.csv'
file_path5 = './data/result/xgboost2_lookback_2.csv'
file_path6 = './data/result/xgboost2_lookback_3.csv'
file_path7 = './data/result/xgboost2_lookback_4.csv'
file_path8 = './data/result/xgboost2_lookback_5.csv'
file_path9 = './data/result/xgboost3_lookback_2.csv'
file_path10 = './data/result/xgboost3_lookback_3.csv'
file_path11 = './data/result/xgboost3_lookback_4.csv'
file_path12 = './data/result/xgboost3_lookback_5.csv'
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.abs(y_true))*100)

if __name__ == "__main__":
    df_final = pd.DataFrame()
    df = pd.DataFrame()

    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)
    df3 = pd.read_csv(file_path3)
    df4 = pd.read_csv(file_path4)

    # Window size 2
    df = df1
    erratic_bool = (df['Demand Pattern'] == 'Erratic')
    erratic_bool_1 = erratic_bool & (df['product_cluster'] == 0)
    erratic_bool_2 = erratic_bool & (df['product_cluster'] == 1)
    erratic_bool_3 = erratic_bool & (df['product_cluster'] == 2)
    erratic_bool_4 = erratic_bool & (df['product_cluster'] == 3)
    erratic_bool_5 = erratic_bool & (df['product_cluster'] == 4)

    intermittent_bool = (df['Demand Pattern'] == 'Intermittent')
    intermittent_bool_1 = intermittent_bool & (df['product_cluster'] == 0)

    lumpy_bool = (df['Demand Pattern'] == 'Lumpy')
    lumpy_bool_1 = lumpy_bool & (df['product_cluster'] == 0)
    lumpy_bool_2 = lumpy_bool & (df['product_cluster'] == 1)
    lumpy_bool_3 = lumpy_bool & (df['product_cluster'] == 2)
    lumpy_bool_4 = lumpy_bool & (df['product_cluster'] == 3)
    lumpy_bool_5 = lumpy_bool & (df['product_cluster'] == 4)

    smooth_bool = (df['Demand Pattern'] == 'Smooth')
    smooth_bool_1 = smooth_bool & (df['product_cluster'] == 0)
    smooth_bool_2 = smooth_bool & (df['product_cluster'] == 1)
    smooth_bool_3 = smooth_bool & (df['product_cluster'] == 2)
    smooth_bool_4 = smooth_bool & (df['product_cluster'] == 3)
    smooth_bool_5 = smooth_bool & (df['product_cluster'] == 4)

    df_final = pd.concat([df_final, df[erratic_bool_1 | erratic_bool_3 | erratic_bool_5 | intermittent_bool_1 | smooth_bool_2 | smooth_bool_4 | smooth_bool_5]], axis=0)

    # Window size 3
    df = df1
    erratic_bool = (df['Demand Pattern'] == 'Erratic')
    erratic_bool_1 = erratic_bool & (df['product_cluster'] == 0)
    erratic_bool_2 = erratic_bool & (df['product_cluster'] == 1)
    erratic_bool_3 = erratic_bool & (df['product_cluster'] == 2)
    erratic_bool_4 = erratic_bool & (df['product_cluster'] == 3)
    erratic_bool_5 = erratic_bool & (df['product_cluster'] == 4)

    intermittent_bool = (df['Demand Pattern'] == 'Intermittent')
    intermittent_bool_1 = intermittent_bool & (df['product_cluster'] == 0)

    lumpy_bool = (df['Demand Pattern'] == 'Lumpy')
    lumpy_bool_1 = lumpy_bool & (df['product_cluster'] == 0)
    lumpy_bool_2 = lumpy_bool & (df['product_cluster'] == 1)
    lumpy_bool_3 = lumpy_bool & (df['product_cluster'] == 2)
    lumpy_bool_4 = lumpy_bool & (df['product_cluster'] == 3)
    lumpy_bool_5 = lumpy_bool & (df['product_cluster'] == 4)

    smooth_bool = (df['Demand Pattern'] == 'Smooth')
    smooth_bool_1 = smooth_bool & (df['product_cluster'] == 0)
    smooth_bool_2 = smooth_bool & (df['product_cluster'] == 1)
    smooth_bool_3 = smooth_bool & (df['product_cluster'] == 2)
    smooth_bool_4 = smooth_bool & (df['product_cluster'] == 3)
    smooth_bool_5 = smooth_bool & (df['product_cluster'] == 4)

    df_final = pd.concat([df_final, df[lumpy_bool_1 | lumpy_bool_4 | smooth_bool_1 | smooth_bool_3]], axis=0)
    # Window size 4
    df = df1
    erratic_bool = (df['Demand Pattern'] == 'Erratic')
    erratic_bool_1 = erratic_bool & (df['product_cluster'] == 0)
    erratic_bool_2 = erratic_bool & (df['product_cluster'] == 1)
    erratic_bool_3 = erratic_bool & (df['product_cluster'] == 2)
    erratic_bool_4 = erratic_bool & (df['product_cluster'] == 3)
    erratic_bool_5 = erratic_bool & (df['product_cluster'] == 4)

    intermittent_bool = (df['Demand Pattern'] == 'Intermittent')
    intermittent_bool_1 = intermittent_bool & (df['product_cluster'] == 0)

    lumpy_bool = (df['Demand Pattern'] == 'Lumpy')
    lumpy_bool_1 = lumpy_bool & (df['product_cluster'] == 0)
    lumpy_bool_2 = lumpy_bool & (df['product_cluster'] == 1)
    lumpy_bool_3 = lumpy_bool & (df['product_cluster'] == 2)
    lumpy_bool_4 = lumpy_bool & (df['product_cluster'] == 3)
    lumpy_bool_5 = lumpy_bool & (df['product_cluster'] == 4)

    smooth_bool = (df['Demand Pattern'] == 'Smooth')
    smooth_bool_1 = smooth_bool & (df['product_cluster'] == 0)
    smooth_bool_2 = smooth_bool & (df['product_cluster'] == 1)
    smooth_bool_3 = smooth_bool & (df['product_cluster'] == 2)
    smooth_bool_4 = smooth_bool & (df['product_cluster'] == 3)
    smooth_bool_5 = smooth_bool & (df['product_cluster'] == 4)

    df_final = pd.concat([df_final, df[erratic_bool_2 | erratic_bool_4 | lumpy_bool_3 | lumpy_bool_5]], axis=0)

    # Window size 5
    df = df1
    erratic_bool = (df['Demand Pattern'] == 'Erratic')
    erratic_bool_1 = erratic_bool & (df['product_cluster'] == 0)
    erratic_bool_2 = erratic_bool & (df['product_cluster'] == 1)
    erratic_bool_3 = erratic_bool & (df['product_cluster'] == 2)
    erratic_bool_4 = erratic_bool & (df['product_cluster'] == 3)
    erratic_bool_5 = erratic_bool & (df['product_cluster'] == 4)

    intermittent_bool = (df['Demand Pattern'] == 'Intermittent')
    intermittent_bool_1 = intermittent_bool & (df['product_cluster'] == 0)

    lumpy_bool = (df['Demand Pattern'] == 'Lumpy')
    lumpy_bool_1 = lumpy_bool & (df['product_cluster'] == 0)
    lumpy_bool_2 = lumpy_bool & (df['product_cluster'] == 1)
    lumpy_bool_3 = lumpy_bool & (df['product_cluster'] == 2)
    lumpy_bool_4 = lumpy_bool & (df['product_cluster'] == 3)
    lumpy_bool_5 = lumpy_bool & (df['product_cluster'] == 4)

    smooth_bool = (df['Demand Pattern'] == 'Smooth')
    smooth_bool_1 = smooth_bool & (df['product_cluster'] == 0)
    smooth_bool_2 = smooth_bool & (df['product_cluster'] == 1)
    smooth_bool_3 = smooth_bool & (df['product_cluster'] == 2)
    smooth_bool_4 = smooth_bool & (df['product_cluster'] == 3)
    smooth_bool_5 = smooth_bool & (df['product_cluster'] == 4)

    df_final = pd.concat([df_final, df[lumpy_bool_2]], axis=0)

    mae = np.round(mean_absolute_error(df_final['Quantity'],df_final['predict_Quantity']), 2)