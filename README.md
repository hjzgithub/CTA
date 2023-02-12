一、研究背景
参考中信期货研报《时序动量与截面动量策略孰优孰劣？》中动量因子的构造方法，通过添加两个过滤条件：
成交量和持仓量，完成时间序列动量策略的构建。

二、数据准备
数据1：25种商品期货主力连续的日线数据，包括开盘价，最高价，最低价和收盘价，成交量和持仓量的数据；
数据2：25种商品期货主力连续的分钟数据，包括开盘价，最高价，最低价和收盘价，成交量和持仓量的数据。
为什么选择这几种？
因为交易比较活跃，是适合交易的品种

三、数据清洗
考虑日间策略，收益计算只采用收盘价。
于是得到收盘价，收益率，成交量和持仓量的日度数据，通过分钟数据得到每日的已实现波动率，存入data.csv

四、因子构建
动量因子 = SML(五日均线)-LML(20日均线)/RV
过滤条件1：成交量变化率
过滤条件2：持仓量变化率
包含参数：5日，10日

五、策略框架
1.趋势反转策略：只在成交量变大和持仓量变小时交易，当动量因子由正变负，做空，当动量因子由负变正，做多
2.组合管理和仓位管理
组合管理的方法是，平均分配资金到每一个品种

交易手数收到资金容量，和仓位风险管理的影响


需要确定
1.可用资金和占用保证金的比例，也就是每个品种的仓位，先不考虑爆仓，也就是每个品种按满仓算
2.以及每次交易动用的可用资金，也就是加仓的深度，设置为5，5即满仓
如果不考虑爆仓，则不需要考虑持仓盈亏，只需要考虑平仓盈亏，平仓盈亏的累加就是总盈亏

输出交易结果

六、回测框架
模型检验：由于是基于先验规则而非基于数据得到后验参数的策略, 所以不需要进行样本外检验
