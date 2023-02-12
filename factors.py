import pandas as pd
import numpy as np

'''
计算每个品种的动量
计算过去五天的移动平均和二十天的移动平均作为SMA和LMA
delta_MA = SMA - LMA
一分钟or五分钟高频数据计算已实现波动率RV
EWMA模型得到当日的预测标准差sigma
动量trend = delta_MA / sigma 作为风险标准化的动量
引入过滤条件，
条件1: 成交量的变化率
条件2: 持仓量的变化率
'''
df = pd.read_csv('data.csv')
# print(df)

def calculate_trend(name):
    df_t = df.copy()
    sma = df_t['return_' + name].rolling(5).mean()
    lma = df_t['return_' + name].rolling(20).mean()
    df_t['trend_' + name] = (sma - lma).shift()
    df_t['power1_' + name] = (df_t['volume_' + name] / df_t['volume_' + name].shift() - 1).shift()
    df_t['power2_' + name] = (df_t['open_interest_' + name] / df_t['open_interest_' + name].shift() - 1).shift()
    return pd.concat([df_factors, df_t['trend_' + name], df_t['power1_' + name], df_t['power2_' + name]], axis=1)

# 得到所有的品种信息
name_list = []
for i in range(1, len(df.columns), 5):
    name_list.append(df.columns[i][6:])
# print(name_list)

# 遍历所有品种
df_factors = df['date']
for name in name_list:
    df_factors = calculate_trend(name)
df_factors.dropna(how='any', axis=0, inplace=True)
df_factors.to_csv('factors.csv', index=0)