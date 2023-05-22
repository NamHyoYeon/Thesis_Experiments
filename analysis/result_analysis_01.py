import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

# metrics for model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

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
file_path13 = './data/result/xgboost4_lookback_2.csv'
file_path14 = './data/result/xgboost4_lookback_3.csv'
file_path15 = './data/result/xgboost4_lookback_4.csv'
file_path16 = './data/result/xgboost4_lookback_5.csv'

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.abs(y_true))*100)

if __name__ == "__main__":
    df = pd.read_csv(file_path13)
    df = df[~df['predict_Quantity'].isna()]
    df['predict_Quantity'] = df['predict_Quantity'].astype(int)

    file_list = [file_path1, file_path2, file_path3, file_path4, file_path5, file_path6, file_path7, file_path8, file_path9, file_path10, file_path11, file_path12, file_path13, file_path14, file_path15, file_path16]
    product_list = []
    product_temp = []

    for f in file_list:
        df_temp = pd.read_csv(f)
        df_temp = df_temp[~df_temp['predict_Quantity'].isna()]
        df_temp = df_temp[df_temp['Quantity'] >= 1]
        df_temp = df_temp[df_temp['predict_Quantity'] >= 1]
        product_list = df_temp['Description'].unique()
        if len(product_temp) == 0:
            product_temp = product_list
        else:
            product_temp = list(set(product_temp) & set(product_list))
    df = df[df['Description'].isin(product_temp)]

    erratic_bool = (df['Demand Pattern'] == 'Erratic')
    erratic_bool_1 = erratic_bool & (df['product_cluster'] == 0)
    erratic_bool_2 = erratic_bool & (df['product_cluster'] == 1)
    erratic_bool_3 = erratic_bool & (df['product_cluster'] == 2)
    erratic_bool_4 = erratic_bool & (df['product_cluster'] == 3)
    erratic_bool_5 = erratic_bool & (df['product_cluster'] == 4)

    intermittent_bool = (df['Demand Pattern'] == 'Intermittent')
    intermittent_bool_1 = intermittent_bool & (df['product_cluster'] == 0)
    intermittent_bool_2 = intermittent_bool & (df['product_cluster'] == 1)
    intermittent_bool_3 = intermittent_bool & (df['product_cluster'] == 2)
    intermittent_bool_4 = intermittent_bool & (df['product_cluster'] == 3)
    intermittent_bool_5 = intermittent_bool & (df['product_cluster'] == 4)

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

    print('Total : ', len(df))
    cnt_total = len(df)
    print('Erratic : ', len(df[erratic_bool]))
    cnt_erratic = len(df[erratic_bool])
    print('Intermittent : ', len(df[intermittent_bool]))
    cnt_intermittent = len(df[intermittent_bool])
    print('Lumpy : ', len(df[lumpy_bool]))
    cnt_lumpy = len(df[lumpy_bool])
    print('Smooth : ', len(df[smooth_bool]))
    cnt_smooth = len(df[smooth_bool])

    # MAE 구하기
    mae = np.round(mean_absolute_error(df['Quantity'],df['predict_Quantity']), 2)
    mae_erratic = np.round(mean_absolute_error(df[erratic_bool]['Quantity'], df[erratic_bool]['predict_Quantity']), 2)
    mae_erratic_1 = np.round(mean_absolute_error(df[erratic_bool_1]['Quantity'], df[erratic_bool_1]['predict_Quantity']), 2)
    mae_erratic_2 = np.round(mean_absolute_error(df[erratic_bool_2]['Quantity'], df[erratic_bool_2]['predict_Quantity']), 2)
    mae_erratic_3 = np.round(mean_absolute_error(df[erratic_bool_3]['Quantity'], df[erratic_bool_3]['predict_Quantity']), 2)
    mae_erratic_4 = np.round(mean_absolute_error(df[erratic_bool_4]['Quantity'], df[erratic_bool_4]['predict_Quantity']), 2)
    mae_erratic_5 = np.round(mean_absolute_error(df[erratic_bool_5]['Quantity'], df[erratic_bool_5]['predict_Quantity']), 2)

    mae_intermittent = np.round(mean_absolute_error(df[intermittent_bool]['Quantity'], df[intermittent_bool]['predict_Quantity']), 2)

    mae_lumpy = np.round(mean_absolute_error(df[lumpy_bool]['Quantity'], df[lumpy_bool]['predict_Quantity']), 2)
    mae_lumpy_1 = np.round(mean_absolute_error(df[lumpy_bool_1]['Quantity'], df[lumpy_bool_1]['predict_Quantity']), 2)
    mae_lumpy_2 = np.round(mean_absolute_error(df[lumpy_bool_2]['Quantity'], df[lumpy_bool_2]['predict_Quantity']), 2)
    mae_lumpy_3 = np.round(mean_absolute_error(df[lumpy_bool_3]['Quantity'], df[lumpy_bool_3]['predict_Quantity']), 2)
    mae_lumpy_4 = np.round(mean_absolute_error(df[lumpy_bool_4]['Quantity'], df[lumpy_bool_4]['predict_Quantity']), 2)
    mae_lumpy_5 = np.round(mean_absolute_error(df[lumpy_bool_5]['Quantity'], df[lumpy_bool_5]['predict_Quantity']), 2)

    mae_smooth = np.round(mean_absolute_error(df[smooth_bool]['Quantity'], df[smooth_bool]['predict_Quantity']), 2)
    mae_smooth_1 = np.round(mean_absolute_error(df[smooth_bool_1]['Quantity'], df[smooth_bool_1]['predict_Quantity']), 2)
    mae_smooth_2 = np.round(mean_absolute_error(df[smooth_bool_2]['Quantity'], df[smooth_bool_2]['predict_Quantity']), 2)
    mae_smooth_3 = np.round(mean_absolute_error(df[smooth_bool_3]['Quantity'], df[smooth_bool_3]['predict_Quantity']), 2)
    mae_smooth_4 = np.round(mean_absolute_error(df[smooth_bool_4]['Quantity'], df[smooth_bool_4]['predict_Quantity']), 2)
    mae_smooth_5 = np.round(mean_absolute_error(df[smooth_bool_5]['Quantity'], df[smooth_bool_5]['predict_Quantity']), 2)

    # RMSE 구하기
    rmse = np.round(np.sqrt(mean_squared_error(df['Quantity'], df['predict_Quantity'])), 2)
    rmse_erratic = np.round(np.sqrt(mean_squared_error(df[erratic_bool]['Quantity'], df[erratic_bool]['predict_Quantity'])), 2)
    rmse_erratic_1 = np.round(np.sqrt(mean_squared_error(df[erratic_bool_1]['Quantity'], df[erratic_bool_1]['predict_Quantity'])), 2)
    rmse_erratic_2 = np.round(np.sqrt(mean_squared_error(df[erratic_bool_2]['Quantity'], df[erratic_bool_2]['predict_Quantity'])), 2)
    rmse_erratic_3 = np.round(np.sqrt(mean_squared_error(df[erratic_bool_3]['Quantity'], df[erratic_bool_3]['predict_Quantity'])), 2)
    rmse_erratic_4 = np.round(np.sqrt(mean_squared_error(df[erratic_bool_4]['Quantity'], df[erratic_bool_4]['predict_Quantity'])), 2)
    rmse_erratic_5 = np.round(np.sqrt(mean_squared_error(df[erratic_bool_5]['Quantity'], df[erratic_bool_5]['predict_Quantity'])), 2)

    rmse_intermittent = np.round(np.sqrt(mean_squared_error(df[intermittent_bool]['Quantity'], df[intermittent_bool]['predict_Quantity'])), 2)

    rmse_lumpy = np.round(np.sqrt(mean_squared_error(df[lumpy_bool]['Quantity'], df[lumpy_bool]['predict_Quantity'])), 2)
    rmse_lumpy_1 = np.round(np.sqrt(mean_squared_error(df[lumpy_bool_1]['Quantity'], df[lumpy_bool_1]['predict_Quantity'])), 2)
    rmse_lumpy_2 = np.round(np.sqrt(mean_squared_error(df[lumpy_bool_2]['Quantity'], df[lumpy_bool_2]['predict_Quantity'])), 2)
    rmse_lumpy_3 = np.round(np.sqrt(mean_squared_error(df[lumpy_bool_3]['Quantity'], df[lumpy_bool_3]['predict_Quantity'])), 2)
    rmse_lumpy_4 = np.round(np.sqrt(mean_squared_error(df[lumpy_bool_4]['Quantity'], df[lumpy_bool_4]['predict_Quantity'])), 2)
    rmse_lumpy_5 = np.round(np.sqrt(mean_squared_error(df[lumpy_bool_5]['Quantity'], df[lumpy_bool_5]['predict_Quantity'])), 2)

    rmse_smooth = np.round(np.sqrt(mean_squared_error(df[smooth_bool]['Quantity'], df[smooth_bool]['predict_Quantity'])), 2)
    rmse_smooth_1 = np.round(np.sqrt(mean_squared_error(df[smooth_bool_1]['Quantity'], df[smooth_bool_1]['predict_Quantity'])), 2)
    rmse_smooth_2 = np.round(np.sqrt(mean_squared_error(df[smooth_bool_2]['Quantity'], df[smooth_bool_2]['predict_Quantity'])), 2)
    rmse_smooth_3 = np.round(np.sqrt(mean_squared_error(df[smooth_bool_3]['Quantity'], df[smooth_bool_3]['predict_Quantity'])), 2)
    rmse_smooth_4 = np.round(np.sqrt(mean_squared_error(df[smooth_bool_4]['Quantity'], df[smooth_bool_4]['predict_Quantity'])), 2)
    rmse_smooth_5 = np.round(np.sqrt(mean_squared_error(df[smooth_bool_5]['Quantity'], df[smooth_bool_5]['predict_Quantity'])), 2)

    # MAPE 구하기
    mape = np.round(mean_absolute_percentage_error(df['Quantity'], df['predict_Quantity']), 2)
    mape_erratic = np.round(mean_absolute_percentage_error(df[erratic_bool]['Quantity'], df[erratic_bool]['predict_Quantity']), 2)
    mape_erratic_1 = np.round(mean_absolute_percentage_error(df[erratic_bool_1]['Quantity'], df[erratic_bool_1]['predict_Quantity']), 2)
    mape_erratic_2 = np.round(mean_absolute_percentage_error(df[erratic_bool_2]['Quantity'], df[erratic_bool_2]['predict_Quantity']), 2)
    mape_erratic_3 = np.round(mean_absolute_percentage_error(df[erratic_bool_3]['Quantity'], df[erratic_bool_3]['predict_Quantity']), 2)
    mape_erratic_4 = np.round(mean_absolute_percentage_error(df[erratic_bool_4]['Quantity'], df[erratic_bool_4]['predict_Quantity']), 2)
    mape_erratic_5 = np.round(mean_absolute_percentage_error(df[erratic_bool_5]['Quantity'], df[erratic_bool_5]['predict_Quantity']), 2)

    mape_intermittent = np.round(mean_absolute_percentage_error(df[intermittent_bool]['Quantity'], df[intermittent_bool]['predict_Quantity']), 2)

    mape_lumpy = np.round(mean_absolute_percentage_error(df[lumpy_bool]['Quantity'], df[lumpy_bool]['predict_Quantity']), 2)
    mape_lumpy_1 = np.round(mean_absolute_percentage_error(df[lumpy_bool_1]['Quantity'], df[lumpy_bool_1]['predict_Quantity']), 2)
    mape_lumpy_2 = np.round(mean_absolute_percentage_error(df[lumpy_bool_2]['Quantity'], df[lumpy_bool_2]['predict_Quantity']), 2)
    mape_lumpy_3 = np.round(mean_absolute_percentage_error(df[lumpy_bool_3]['Quantity'], df[lumpy_bool_3]['predict_Quantity']), 2)
    mape_lumpy_4 = np.round(mean_absolute_percentage_error(df[lumpy_bool_4]['Quantity'], df[lumpy_bool_4]['predict_Quantity']), 2)
    mape_lumpy_5 = np.round(mean_absolute_percentage_error(df[lumpy_bool_5]['Quantity'], df[lumpy_bool_5]['predict_Quantity']), 2)

    mape_smooth = np.round(mean_absolute_percentage_error(df[smooth_bool]['Quantity'], df[smooth_bool]['predict_Quantity']), 2)
    mape_smooth_1 = np.round(mean_absolute_percentage_error(df[smooth_bool_1]['Quantity'], df[smooth_bool_1]['predict_Quantity']), 2)
    mape_smooth_2 = np.round(mean_absolute_percentage_error(df[smooth_bool_2]['Quantity'], df[smooth_bool_2]['predict_Quantity']), 2)
    mape_smooth_3 = np.round(mean_absolute_percentage_error(df[smooth_bool_3]['Quantity'], df[smooth_bool_3]['predict_Quantity']), 2)
    mape_smooth_4 = np.round(mean_absolute_percentage_error(df[smooth_bool_4]['Quantity'], df[smooth_bool_4]['predict_Quantity']), 2)
    mape_smooth_5 = np.round(mean_absolute_percentage_error(df[smooth_bool_5]['Quantity'], df[smooth_bool_5]['predict_Quantity']), 2)

    print(df.groupby(['Demand Pattern']).count())
    print(df.groupby(['Demand Pattern','product_cluster']).count())

    # # RMSE 구하기
    # rmse = np.round(np.sqrt(mean_squared_error(df['Quantity'], df['predict_Quantity'])), 2)
    # rmse_erratic = np.round(np.sqrt(mean_squared_error(df[erratic_bool]['Quantity'], df[erratic_bool]['predict_Quantity'])), 2)
    # rmse_intermittent = np.round(np.sqrt(mean_squared_error(df[intermittent_bool]['Quantity'], df[intermittent_bool]['predict_Quantity'])), 2)
    # rmse_lumpy = np.round(np.sqrt(mean_squared_error(df[lumpy_bool]['Quantity'], df[lumpy_bool]['predict_Quantity'])), 2)
    # rmse_smooth = np.round(np.sqrt(mean_squared_error(df[smooth_bool]['Quantity'], df[smooth_bool]['predict_Quantity'])), 2)

