import numpy as np
import pandas as pd
import akshare as ak
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential  # vscode 解析器问题
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping


# 训练品种和时长/分钟
train_type = "EB2505"
train_period = "60"

# 数据加载与预处理
data = ak.futures_zh_minute_sina(symbol=train_type, period=train_period)
data['return'] = data['close'].pct_change()
data = data.dropna()

# 划分训练集和测试集（时间序列需按顺序划分）
split = int(0.8 * len(data))
train_data = data.iloc[:split]
test_data = data.iloc[split:]


#print(f"train_data:{train_data}")
print(f"test_data:{test_data}")


# 对训练集和测试集分别归一化（避免数据泄露）
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(
    train_data[['open', 'high', 'low', 'close', 'volume', 'hold']])
scaled_test = scaler.transform(
    test_data[['open', 'high', 'low', 'close', 'volume', 'hold']])

# 生成时序特征和标签
def create_dataset(data, look_back=60):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, :])
        y.append(1 if (data[i, 3] > data[i-look_back, 3])
                 else 0)  # 比较当前close与look_back前的close
    return np.array(X), np.array(y)


X_train, y_train = create_dataset(scaled_train)
X_test, y_test = create_dataset(scaled_test)

# 检查标签分布
print("训练集标签分布:", np.unique(y_train, return_counts=True))
print("测试集标签分布:", np.unique(y_test, return_counts=True))

# 若存在严重不平衡，可添加类别权重
class_weights = compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train)
class_weights = {0: class_weights[0], 1: class_weights[1]}

model = Sequential()
model.add(LSTM(units=64, return_sequences=True,
          input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall'])


# 添加早停法
early_stop = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weights,  # 若存在不平衡
    callbacks=[early_stop]
)

# 使用混淆矩阵和分类报告
y_pred = (model.predict(X_test) > 0.5).astype(int)

'''
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
'''

print(f"y_pred:{y_pred}")

# 绘制训练曲线
'''
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
'''
