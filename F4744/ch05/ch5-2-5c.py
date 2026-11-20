import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Input, Dense

np.random.seed(10)  # 指定亂數種子
# 載入糖尿病資料集
df = pd.read_csv("./diabetes.csv")
dataset = df.values
np.random.shuffle(dataset)  # 使用亂數打亂資料
# 分割成特徵資料和標籤資料
X = dataset[:, 0:8]
y = dataset[:, 8]
# 特徵標準化
X -= X.mean(axis=0)
X /= X.std(axis=0)
# 分割訓練和測試資料集
X_train, y_train = X[:690], y[:690]     # 訓練資料前690筆
X_test, y_test = X[690:], y[690:]       # 測試資料後78筆
# 定義模型
model = Sequential()
model.add(Input(shape=(8,)))
model.add(Dense(8, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
# 編譯模型
model.compile(loss="binary_crossentropy", optimizer="adam", 
              metrics=["accuracy"])
# 訓練模型
model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=0)
# 評估模型
loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
print("訓練資料集的準確度 = {:.2f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("測試資料集的準確度 = {:.2f}".format(accuracy))
print("--------------------------")
# 測試資料集的預測值
y_pred = model.predict(X_test, batch_size=10, verbose=0)

import tensorflow as tf

y_pred_classes = tf.cast(y_pred > 0.5, dtype=tf.int32)
print(y_pred_classes[0].numpy(), y_pred_classes[1].numpy())




