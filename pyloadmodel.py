import numpy as np
import pandas as pd
import akshare as ak
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential    #vscode 解析器问题
from tensorflow.keras.layers import LSTM, Dense


# 数据加载与预处理
data =ak.futures_zh_minute_sina(symbol="M2505", period="60")
data['return'] = data['close'].pct_change()
data = data.dropna()

#print(data)


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['open', 'high', 'low', 'close', 'volume','hold','return']])

arr = np.array([[1,2,3,4,5,6,-1],[2,2,3,4,5,6,1],[3,2,3,4,5,6,-1],
    [4,2,3,4,5,6,-1],[5,2,3,4,5,6,1],[6,2,3,4,5,6,-1],
    [7,2,3,4,5,6,-1],[8,2,3,4,5,6,-1],[9,2,3,4,5,6,1],
    [10,2,3,4,5,6,1],[11,2,3,4,5,6,-1],[12,2,3,4,5,6,1]])

print(f"arr:{arr}")

# 特征与标签
X = []
y = []
for i in range(3, len(arr)):
    X.append(arr[i-3:i, :-1])
    y.append(1 if arr[i, -1] > 0 else 0)
X, y = np.array(X), np.array(y)
print(f"X:{X}")
print(f"y:{y}")

# 划分训练集和测试集
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"X_train:{X_train}")
print(f"X_test:{X_test}")
print(f"y_train:{y_test}")
print(f"y_test:{y_test}")





# 构建LSTM模型
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=20, batch_size=32)

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.2f}')
print(f'Test loss: {loss:.2f}')


# 预测
predictions = model.predict(X_test)
#predictions = (predictions > 0.5).astype(int)
print(predictions)
