from strategy import TrendStrategy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 条件初始化
CASH = 10000000
with open("test.txt", "r") as f:
    good_list = f.read().splitlines()

N = len(good_list)
INTEREST_RATE = 0.05
POSITION = 0

INFO = pd.read_csv('info.csv', index_col=0)
NAMES = INFO.index
UNITS = INFO['unit']
FEES = INFO['fee']
MARGINS = INFO['margin']

DATA = pd.read_csv('data.csv').iloc[250:, :]
DATA['date'] = DATA['date'].astype('datetime64[D]')
FACTORS = pd.read_csv('factors.csv')
FACTORS['date'] = FACTORS['date'].astype('datetime64[D]')

# 遍历各个品种
def get_portfolio(sample_name):

    num_total = 0
    times_list = []
    win_rate_list = []
    pnl_ratio_list = []

    for name in good_list:
        tstrategy = TrendStrategy(DATA, FACTORS, name, CASH/N, POSITION, UNITS, FEES, MARGINS)
        df_result = tstrategy.get_trading()
        cash_profit = df_result['cash_profit']

        if num_total == 0:
            cash_all = pd.DataFrame(df_result['cash']) 
            cash_all.index = df_result['date'] 
            cash_all.columns = [name]
            guarantee_total = pd.Series(df_result['guarantee']) 
        else:
            df_new = pd.DataFrame(df_result['cash'])
            df_new.index = df_result['date'] 
            df_new.columns = [name]
            cash_all = pd.concat([cash_all, df_new], axis=1)
            guarantee_total += pd.Series(df_result['guarantee'])
        
        # 得到交易次数，胜率和盈亏比的统计结果

        win_times = 0
        profit = 0
        lose_times = 0
        loss = 0
        for i in cash_profit.values:
            if i > 0:
                win_times += 1
                profit += i
            elif i < 0:
                lose_times += 1
                loss += -i
        
        total_times = win_times + lose_times
        if win_times > 0 and lose_times > 0:
            win_rate = win_times / total_times
            pnl_ratio = (profit/win_times) / (loss/lose_times)
            times_list.append(total_times)
            win_rate_list.append(win_rate)
            pnl_ratio_list.append(pnl_ratio)
        num_total += 1

        if name == sample_name:
            t = df_result['date'].astype('datetime64[D]')
            x1 = df_result['close'] 
            x2 = df_result['position']

            #绘制第一个Y轴
            fig = plt.figure(figsize=(20,8), dpi=80)
            ax = fig.add_subplot(111)
            lin1 = ax.plot(t, x1, label="close price")
            
            #绘制另一Y轴    
            ax1 = ax.twinx()
            lin2 = ax1.plot(t, x2, color="red", label="position")
            
            #合并图例
            lins = lin1 + lin2
            labs = [l.get_label() for l in lins]
            ax.legend(lins, labs, loc="upper left", fontsize=15)
            plt.show()

            x3 = df_result['cash_profit']
            x4 = df_result['account_profit']
            plt.xticks(rotation=30)
            plt.plot(t, x3, label='cash profit')
            plt.plot(t, x4, color='y', label='account profit') 
            plt.legend()
            plt.show()

            x5 = df_result['cash']
            plt.xticks(rotation=30)
            plt.plot(t, x5, label='cash')
            plt.legend()
            plt.show()

            x6 = df_result['guarantee'] / df_result['cash']
            plt.xticks(rotation=30)
            plt.plot(t, x6, label='occupation ratio of gurantee')
            plt.legend()
            plt.show()

    return times_list, win_rate_list, pnl_ratio_list, cash_all, guarantee_total

def MDD(data):
    N = len(data) 
    DD = np.zeros((N-1, N-1)) 
    for i in range(N-1):
        v_i = data[i]                     # 第i个交易日的基金净值
        for j in range(i+1, N):
            v_j = data[j]                 # 第j个交易日的基金净值
            DD[i,j-1] = (v_i - v_j) / v_i # 从第i个交易日到第j个交易日期间的基金期间回撤率
    Max_DD = DD.max()
    m, n = DD.shape
    index = int(DD.argmax())
    x = int(index / n)
    y = index % n
    start_loss_max = data.index[x]
    end_loss_max = data.index[y+1]
    return start_loss_max, end_loss_max, Max_DD

def result_analysis(report_path, sample_name=0):
    times_list, win_rate_list, pnl_ratio_list, cash_all, guarantee_total = get_portfolio(sample_name)
    times_mean = np.array(times_list).mean()
    win_rate_mean = np.array(win_rate_list).mean()

    start_date = FACTORS['date'].iloc[0] if FACTORS['date'].iloc[0] >= DATA['date'].iloc[0] else DATA['date'].iloc[0]
    end_date = FACTORS['date'].iloc[-1] if FACTORS['date'].iloc[-1] <= DATA['date'].iloc[-1] else DATA['date'].iloc[-1]

    cash_total = cash_all.sum(axis=1)
    cash_total.index = DATA['date']

    # 计算交易期间年化收益率
    day_total = len(cash_total.loc[start_date:end_date])
    annualized_rate = (cash_total.loc[end_date] / cash_total.loc[start_date] - 1)/ day_total * 250
    
    # 计算交易期间等权组合净值曲线
    value_net = cash_total.loc[start_date:end_date] / cash_total.loc[start_date]

    # 计算交易期间最大回撤
    start_loss_max, end_loss_max, Max_DD = MDD(value_net)
    return_net = ((value_net - value_net.shift()) / value_net.shift()).dropna()

    # 画出收益率的时序图
    return_net.plot()
    plt.title('time series of return rate')
    plt.savefig(report_path + '1.png')
    plt.show()
    
    plt.hist(return_net, bins=50)
    plt.title('distribution of return rate')
    plt.savefig(report_path + '2.png')
    plt.show()

    # 计算夏普比率
    sharpe_ratio = (return_net.mean() * 250 - INTEREST_RATE) / (return_net.std(ddof = 1) * np.sqrt(250))

    print('\n回测结果分析:\n')
    print('交易品种:', good_list)
    print('起始资金:', int(cash_total.loc[start_date]))
    print('结束资金:', int(cash_total.loc[end_date]))
    print('起始日期:', start_date.strftime("%Y-%m-%d"))
    print('结束日期:', end_date.strftime("%Y-%m-%d"))
    print('总交易日数:', day_total)
    print('年化收益率: {:.2f}%'.format(annualized_rate*100))
    print('最大回撤: {:.2f}%'.format(Max_DD*100))
    print('最大回撤出现时间:', 'from ' + start_loss_max.strftime("%Y-%m-%d") + ' to ' + end_loss_max.strftime("%Y-%m-%d"))
    print('卡玛比率: {:.2f}'.format(annualized_rate/Max_DD))
    print('夏普比率: {:.2f}'.format(sharpe_ratio))
    print('平均交易天数:', round(times_mean))
    print('平均胜率: {:.2f}%'.format(win_rate_mean*100))

    guarantee_total.index = DATA['date']
    cash_total.index = DATA['date']

    # 查看保证金占用比例
    t = cash_total.loc[start_date:end_date].index
    gt = guarantee_total.loc[start_date:end_date] / cash_total.loc[start_date:end_date]
    plt.plot(t, gt)
    plt.xticks(rotation=30)
    plt.title('occupation ratio of gurantee')
    plt.savefig(report_path + '3.png')
    plt.show()
    

    # 画出等权组合净值曲线
    vt = cash_total.loc[start_date:end_date] / cash_total.loc[start_date]
    plt.plot(t, vt)
    plt.xticks(rotation=30)
    plt.title('net value of portfolio')
    plt.savefig(report_path + '4.png')
    plt.show()

    # 各品种净值曲线
    for i in cash_all.columns:
        v = cash_all.loc[start_date:end_date, i] / cash_all.loc[start_date, i]
        plt.plot(t, v)

    plt.xticks(rotation=30)
    plt.title('net value of given future species')
    plt.savefig(report_path + '5.png')
    plt.show()

    # 画出胜率和盈亏比的散点图
    x = win_rate_list
    y = pnl_ratio_list  
    sc = plt.scatter(x, y, c=times_list, cmap='YlGnBu')  
    plt.colorbar(sc)  
    plt.xlabel('win rate')
    plt.ylabel('pnl ratio')
    plt.savefig(report_path + '6.png')
    plt.show()

result_analysis('report/')