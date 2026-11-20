import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Input, Dense
import warnings
warnings.filterwarnings("ignore")

np.random.seed(7)
# 載入Titanic的訓練和測試資料集
df_train = pd.read_csv("./titanic_train.csv")
df_test = pd.read_csv("./titanic_test.csv")
dataset_train = df_train.values.astype('float32')
dataset_test = df_test.values.astype('float32')
# 分割成特徵資料和標籤資料
X_train = dataset_train[:, 0:9]
y_train = dataset_train[:, 9]
X_test = dataset_test[:, 0:9]
y_test = dataset_test[:, 9]
# 特徵標準化
X_train -= X_train.mean(axis=0)
X_train /= X_train.std(axis=0)
X_test -= X_test.mean(axis=0)
X_test /= X_test.std(axis=0)
# 定義模型
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(11, activation="relu"))
model.add(Dense(11, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
# 編譯模型
model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=["accuracy"])
# 訓練模型
print("Training ...")
model.fit(X_train, y_train, epochs=18, batch_size=10, verbose=0)
# 評估模型
print("\nTesting ...")
loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
print("訓練資料集的準確度 = {:.2f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("測試資料集的準確度 = {:.2f}".format(accuracy))
# 儲存Keras模型
print("Saving Model: titanic.keras ...")
model.save("titanic.keras")