import pandas as pd
import matplotlib.pyplot as plt
import datetime

file_path = './data/Online Retail Dataset.csv'

if __name__ == "__main__":
    df = pd.read_csv(file_path)
    df['UnitPrice'] = df['UnitPrice'].astype('int')
    df['CustomerID'] = df['CustomerID'].astype('str')

    print(df.info())

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['YYYY'] = df['InvoiceDate'].dt.year
    df['MM'] = df['InvoiceDate'].dt.month
    df['DD'] = df['InvoiceDate'].dt.day
    df['YYYYMM'] = df['YYYY'].astype('str') + df['MM'].astype('str').apply(lambda x: str(x).zfill(2))

    # 2010/01 ~ 2011/08  1년 8개월 치 데이터
    print(df['YYYYMM'].unique())

    # 전체 건수 542,014
    print(df.info())

    # Description(상품명) unique 4221 개..
    print(len(df['Description'].unique()))

    # CustomerID(고객번호) unique 4373 개..
    print(len(df['CustomerID'].unique()))

    # Description(상품명) 별 주문 청구 횟수
    df2 = df.groupby('Description')['InvoiceNo'].count().reset_index()
    df2.reset_index(drop=True, inplace=True)
    df2.rename(columns={'InvoiceNo':'CNT'}, inplace=True)

    # 상품 주문청구 횟수가 100 개 이상 중 TOP10
    tgt_df = df2[df2['CNT'] > 100].sort_values('CNT', ascending=False)[0:10]

    # 상품별 월별 주문 수량 PLOT
    for index, row in tgt_df.iterrows():
        plt.title(row['Description'] + '(Invoice Count : {})'.format(row['CNT']))
        print(row['Description'])
        # Quantity > 0 환불 데이터 제외..
        tf1 = df[(df['Description'] == row['Description']) & (df['Quantity'] > 0)]
        tf1.plot(x='YYYYMM', y='Quantity')
        plt.show()