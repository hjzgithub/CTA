import os
import pandas as pd

def data_clean(path_start, path_end1, path_end2):
    files = os.listdir(path_start)
    
    for file in files:
        df_file = pd.read_csv(path_start + '/' + file)
        data_type = file[-7:-4]
        name = file[:-11]
        df_result = pd.DataFrame()
        if data_type == 'day':
            df_result['date'] = pd.to_datetime(df_file.iloc[:, 4].astype('str'))
            df_result[['open', 'high', 'low', 'close']] = df_file.iloc[:, :4]
            df_result[['volume', 'open_interest']] = df_file.iloc[:, -2:]
            df_result.to_csv(path_end1 + '/' + name + '.csv', index=0)
        elif data_type == 'min':
            df_result['date'] = pd.to_datetime(df_file.iloc[:, 0].astype('str'))
            df_result['time'] = df_file.iloc[:, 1]
            df_result[['open', 'high', 'low', 'close', 'volume', 'open_interest']] = df_file.iloc[:, -6:]
            df_result.to_csv(path_end2 + '/' + name + '.csv', index=0)
            
data_clean('原始数据', '期货日线数据', '期货分钟数据')

