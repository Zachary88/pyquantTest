from jqdata import *

# 初始化函数，设定基准等
def initialize(context):
    # 设定沪深300作为基准
    set_benchmark('000300.XSHG')
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    
    # 股票代码（这里以平安银行为例）
    context.security = '000001.XSHE'
    
    # 设置策略参数
    context.N = 9  # KDJ指标的周期
    context.M = 3  # KDJ指标的平滑周期
    
    # 设置初始资金
    context.cash = 100000
    
    # 设置交易费用（根据实际情况调整）
    set_order_cost(OrderCost(open_tax=0, 
                            close_tax=0.001, 
                            open_commission=0.0003, 
                            close_commission=0.0003, 
                            close_today_commission=0, 
                            min_commission=5), 
                  type='stock')

# 每个单位时间(如果按天回测,则每天调用一次)调用一次
def handle_bar(context, bar_dict):
    security = context.security
    
    # 获取历史数据
    prices = get_bars(security, count=context.N*3, 
                      unit='1d', fields=['close','high','low'])
    
    # 计算KDJ指标
    K, D, J = calculate_KDJ(prices, context.N, context.M)
    
    # 获取当前和前一天的J值
    current_j = J[-1]
    previous_j = J[-2] if len(J) >= 2 else 0
    
    # 获取当前仓位
    current_position = context.portfolio.positions[security].total_amount
    
    # 交易逻辑：当J值由正转负且当前无持仓时买入
    if previous_j >= 0 and current_j < 0:
        if current_position == 0:
            # 全仓买入
            order_value(security, context.cash)
            log.info("买入 %s 于 %s" % (security, current_j))
            
def calculate_KDJ(prices, N=9, M=3):
    """
    计算KDJ指标
    返回：K, D, J 值列表
    """
    low_list = prices.low.rolling(N).min()
    high_list = prices.high.rolling(N).max()
    rsv = (prices.close - low_list) / (high_list - low_list) * 100
    
    K = rsv.ewm(alpha=1/M, adjust=False).mean()  # 使用EMA计算K值
    D = K.ewm(alpha=1/M, adjust=False).mean()    # 使用EMA计算D值
    J = 3 * K - 2 * D
    
    return K.tolist(), D.tolist(), J.tolist()