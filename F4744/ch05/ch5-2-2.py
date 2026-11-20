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
# 定義模型
model = Sequential()
model.add(Input(shape=(8,)))
model.add(Dense(10, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()  # 顯示模型摘要資訊
print("--------------------------")
# 編譯模型
model.compile(loss="binary_crossentropy", optimizer="sgd",
              metrics=["accuracy"])
# 訓練模型
model.fit(X, y, epochs=150, batch_size=10)
print("--------------------------")
# 評估模型
loss, accuracy = model.evaluate(X, y)
print("準確度 = {:.2f}".format(accuracy))

