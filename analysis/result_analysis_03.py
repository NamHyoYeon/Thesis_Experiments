import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import datetime

import scipy.stats
if __name__ == "__main__":
    df = pd.DataFrame()

    df['before'] = [38.91 ,163.2 ,116.16, 134.50]
    df['after'] = [38.60 ,163.0 ,116.01, 134.22]
    df['diff']  = df['before'] - df['after']

    print(df)
    # T-TEST 양측 검정, 대응표본 t 검증
    scipy.stats.ttest_rel(df['before'], df['after'])
