import os
import pandas as pd
import numpy as np

def get_data(file_end):
    # path为工作区内目标文件夹的相对路径
    path = '期货日线数据'
    files = os.listdir(path)

    # 得到每个品种的daily return, 存在一张表中
    initial_file = 'ag.csv'
    df = pd.DataFrame()
    df['date'] = pd.read_csv(path + '/' + initial_file)['date']
    # print(df)

    # 首先得到每个品种的close price与return
    for file in files:
        df_file = pd.read_csv(path + '/' + file)
        name = file[:-4]
        df_file['close_' + name] = df_file['close']
        df_file['return_' + name] = df_file['close'] / df_file['close'].shift() - 1
        df_file['volume_' + name] = df_file['volume'] 
        df_file['open_interest_' + name] = df_file['open_interest']
        # print(df_file)
        df_file_min = pd.read_csv('期货分钟数据/' + file)
        df_file_min['return_' + name] = (df_file_min['close'] / df_file_min['close'].shift()).apply(np.log)
        df_file_min['return**2_' + name] = df_file_min['return_' + name] ** 2
        df_vol = df_file_min.groupby('date').aggregate({'return**2_' + name : np.sum}).reset_index()
        df_vol.columns = ['date', 'rv_' + name]
        df = pd.merge(df, df_file[['date', 'close_' + name, 'return_' + name, 'volume_' + name, 'open_interest_' + name]], on='date', how='outer')
        df = pd.merge(df, df_vol, on='date', how='left')
    df_result = df.iloc[1:, :].dropna(how='any', axis=0)
    # print(df_result)
    df_result.to_csv(file_end, index=0)

get_data('data.csv')