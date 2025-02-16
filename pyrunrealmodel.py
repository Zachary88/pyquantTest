import numpy as np
import pandas as pd
import akshare as ak
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


# 加载训练模型
model = load_model("FG2505_lstm_model20250216_154517.h5")

# 实盘品种和时长/分钟
train_type = "FG2505"
train_period = "120"

# 加载实盘数据
data = ak.futures_zh_minute_sina(symbol=train_type, period=train_period)
data['return'] = data['close'].pct_change()
data = data.dropna()
print(f"data:{data}")

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(
    data[['open', 'high', 'low', 'close', 'volume', 'hold', 'return']])

# 获得实盘数据
X = []
X.append(scaled_data[-60:, :-1])
X = np.array(X)

# 预测涨跌
predictions = model.predict(X)
now = datetime.now()
print(f"{now}:预测{train_type}品种{train_period}分钟上涨概率：{predictions}")
