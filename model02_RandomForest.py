import pandas as pd
import numpy as np
import datetime

# plot packages
import matplotlib.pyplot as plt

# model packages
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# metrics for model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

file_path = './data/df_final.csv'

if __name__ == "__main__":
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
    print(df_tgt.head())
    df_tgt = df_tgt.set_index('YYYYMM')

    print(df_tgt)

    # Description(상품) 별로 예측, 가장 마지막 달을 기준으로 train
    product_list = df_tgt['Description'].unique()
    df_tgt['predict_Quantity'] = np.nan

    print(df_tgt.info())

    bf = datetime.now()

    # Generate some random data for regression
    X, y = make_regression(n_samples=1000, n_features=10, random_state=42)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest regressor
    reg = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the regressor to the training data
    reg.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = reg.predict(X_test)

    # Evaluate the performance of the regressor
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    af = datetime.now()