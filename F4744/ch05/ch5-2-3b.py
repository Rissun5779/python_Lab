import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Input, Dense
from keras.utils import to_categorical

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
# One-hot編碼
y = to_categorical(y)
# 定義模型
model = Sequential()
model.add(Input(shape=(8,)))
model.add(Dense(10, kernel_initializer="random_uniform", 
                bias_initializer="ones",
                activation="relu"))
model.add(Dense(8, kernel_initializer="random_uniform",
                bias_initializer="ones",
                activation="relu"))
model.add(Dense(2, kernel_initializer="random_uniform",
                bias_initializer="ones",
                activation="softmax"))
# model.summary()   # 顯示模型摘要資訊
# 編譯模型
model.compile(loss="binary_crossentropy", optimizer="sgd", 
              metrics=["accuracy"])
# 訓練模型
model.fit(X, y, epochs=150, batch_size=10, verbose=0)
# 評估模型
loss, accuracy = model.evaluate(X, y, verbose=0)
print("準確度 = {:.2f}".format(accuracy))

