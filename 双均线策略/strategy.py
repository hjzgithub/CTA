import pandas as pd

class TrendStrategy:
    def __init__(self, df_data, df_factors, name, cash, position, units, fees, margins):
        self.df_data = df_data
        self.df_factors = df_factors
        self.name = name
        self.cash = cash
        self.position = position
        self.units = units
        self.fees = fees
        self.margins = margins

    def get_data(self):
        df_data = self.df_data
        data_close = pd.Series(list(df_data['close_' + self.name]), index=df_data['date'])
        return data_close

    def get_factors(self):
        name = self.name
        factors = self.df_factors[['trend_' + name, 'power1_' + name, 'power2_' + name]]
        factors.index = self.df_factors['date']
        return factors
        
    def get_orders(self):
        name = self.name
        factors = self.get_factors()

        k = factors.index
        orders = {}
        for i in range(1, len(k)):
            trend = factors.loc[k[i], 'trend_' + name]
            power1 = factors.loc[k[i], 'power1_' + name]
            power2 = factors.loc[k[i], 'power2_' + name]
            
            if trend < 0 < factors.loc[k[i-1], 'trend_' + name] and power1 > 0 and power2 < 0:
                orders[k[i]] = -1

            elif trend > 0 > factors.loc[k[i-1], 'trend_' + name] and power1 > 0 and power2 < 0:
                orders[k[i]] = 1 

        return orders

    def get_hands(self, cash, price, margin, occupation_ratio=0.75, depth=10):
        unit = self.units[self.name]
        n = 0
        n_max = int(cash / (margin * price * unit) * occupation_ratio) # 满仓最大乘以保证金占用率
        if n_max > 0: 
            for k in range(n_max):
                if n_max / (k+1) <= depth:
                    n = k+1
                    break
        return n, n_max

    # 根据下单命令，得到每日的仓位，现金，平仓盈亏，持仓盈亏，净值等数据
    def get_trading(self):

        position, cash, margin = self.position, self.cash, self.margins[self.name]

        data_close = self.get_data()
        dates = data_close.index

        orders = self.get_orders()
        
        unit = self.units[self.name]
        position_list = []
        cash_list = []
        cash_profit_list = []
        account_profit_list = []
        guarantee_list = []
        open_price = 0 # 初始化开仓价
         
        for i in range(len(dates)):
            date = dates[i]
            close_price = data_close[date]

            cash_profit = 0 # 初始化当日平仓盈亏
            account_profit = 0 # 初始化当日账面盈亏
            delta_position = 0 # 初始化调仓数

            # 三种情况：处在交易指令那天，达到强制平仓要求，回测最后一天平仓
            if date in orders and cash > 0.3 * self.cash:
                if i > 0:
                    last_close_price = data_close[dates[i-1]]
                else:
                    last_close_price = data_close[date]
                n, n_max = self.get_hands(cash, last_close_price, margin) 
                
                delta_position = orders[date] * n # 交易手数
                if position + delta_position < -n_max or position + delta_position > n_max:
                    if delta_position > 0:
                        delta_position = n_max - position
                    elif delta_position < 0:
                        delta_position = -n_max - position
        
            if cash <= 0.3 * self.cash or i == len(dates) - 1:
                delta_position = -position
            
            # 执行开平仓操作
            if delta_position != 0:
                new_position = position + delta_position
                if position == 0:
                    # 买入开仓 or 卖出开仓
                    open_price = close_price
                elif position > 0:
                    if delta_position > 0:
                        # 买入加仓, 此时开仓价变成均价
                        open_price = (open_price * position + close_price * delta_position) / new_position
                    elif delta_position < 0:
                        if new_position >= 0:
                            # 卖出平仓
                            cash_profit = (close_price - open_price) * (-delta_position) * unit
                        elif new_position < 0:
                            # 卖出平仓 +  卖出开仓
                            cash_profit = (close_price - open_price) * position * unit
                            open_price = close_price
                elif position < 0:
                    if delta_position < 0:
                        # 卖出加仓
                        open_price = (open_price * position + close_price * delta_position) / new_position
                    elif delta_position > 0:
                        if new_position <= 0:
                        # 买入平仓
                            cash_profit = (close_price - open_price) * (-delta_position) * unit
                        elif new_position > 0:
                            # 买入平仓 + 买入开仓
                            cash_profit = (close_price - open_price) * position * unit 
                            open_price = close_price
                position = new_position
                cash_profit -= self.fees[self.name] * abs(delta_position) # 扣除交易手续费
                cash += cash_profit
            account_profit = (close_price - open_price) * position * unit  # 计算当日持仓盈亏
            guarantee = margin * close_price * abs(position) * unit # 更新占用保证金

            position_list.append(position)
            cash_list.append(cash)
            cash_profit_list.append(cash_profit)
            account_profit_list.append(account_profit)
            guarantee_list.append(guarantee)

        df_result = pd.DataFrame(dates)
        df_result['close'] = data_close.values
        df_result['position'] = position_list # 持仓(单位:手)
        df_result['account_profit'] = account_profit_list # 持仓盈亏
        df_result['cash_profit'] = cash_profit_list # 平仓盈亏
        df_result['cash'] = cash_list
        df_result['guarantee'] = guarantee_list # 每日保证金占用
        
        return df_result