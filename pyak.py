import time 
import pandas as pd
import akshare as ak
import os

def clear_print():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_realtime_future():
    '''获得实时数据'''
    df=ak.futures_zh_spot(symbol="P2505,FG2505,V2505", market="CF", adjust="0")
    return df

while True:
    try:
        clear_print()
        data = get_realtime_future()
        print(f"\n当前时间: {pd.Timestamp.now()}")
        print(data[['symbol','high','low','time','open','current_price']])
        time.sleep(10)  # 10秒间隔
    except Exception as e:
        print(f"获取数据失败: {e}")
        time.sleep(10)

