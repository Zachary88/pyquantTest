import time 
import pandas as pd
import akshare as ak
import os

def clear_print():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_realtime_future():
    '''获得实时数据'''
    df=ak.futures_zh_spot(symbol="V2505,P2505,EB2503,FG2505", market="CF", adjust="0")
    return df

while True:
    try:
        data = get_realtime_future()
        print(f"\n当前时间: {pd.Timestamp.now()}")
        print(data[['symbol','high','low','time','open','current_price']])
        time.sleep(10)  # 5秒间隔
        clear_print()
    except Exception as e:
        print(f"获取数据失败: {e}")
        time.sleep(10)

