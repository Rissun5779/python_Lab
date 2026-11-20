import numpy as np
from keras.datasets import mnist
from keras import Sequential
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import to_categorical

# 指定亂數種子
np.random.seed(7)
# 載入資料集
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# 建立2個資料集，一個數字小於 5，一個數字大於等於 5
X_train_lt5 = X_train[y_train < 5]
y_train_lt5 = y_train[y_train < 5]
X_test_lt5 = X_test[y_test < 5]
y_test_lt5 = y_test[y_test < 5]

X_train_gte5 = X_train[y_train >= 5]
y_train_gte5 = y_train[y_train >= 5] - 5
X_test_gte5 = X_test[y_test >= 5]
y_test_gte5 = y_test[y_test >= 5] - 5
# 將圖片轉換成 4D 張量
X_train_lt5 = X_train_lt5.reshape(
        (X_train_lt5.shape[0], 28, 28, 1)).astype("float32")
X_test_lt5 = X_test_lt5.reshape(
        (X_test_lt5.shape[0], 28, 28, 1)).astype("float32")
X_train_gte5 = X_train_gte5.reshape(
        (X_train_gte5.shape[0], 28, 28, 1)).astype("float32")
X_test_gte5 = X_test_gte5.reshape(
        (X_test_gte5.shape[0], 28, 28, 1)).astype("float32")
# 因為是固定範圍, 所以執行正規化, 從 0-255 至 0-1
X_train_lt5 = X_train_lt5 / 255
X_test_lt5 = X_test_lt5 / 255
X_train_gte5 = X_train_gte5 / 255
X_test_gte5 = X_test_gte5 / 255
# One-hot編碼
y_train_lt5 = to_categorical(y_train_lt5, 5)
y_test_lt5 = to_categorical(y_test_lt5, 5)
y_train_gte5 = to_categorical(y_train_gte5, 5)
y_test_gte5 = to_categorical(y_test_gte5, 5)
# 定義模型
model = Sequential()
model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(8, kernel_size=(3, 3),
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(8, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(5, activation="softmax"))
model.summary()   # 顯示模型摘要資訊
print("--------------------------")
# 編譯模型
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])
# 訓練模型
history = model.fit(X_train_lt5, y_train_lt5, validation_split=0.2,
                    epochs=5, batch_size=128, verbose=2)
print("--------------------------")
# 評估模型
loss, accuracy = model.evaluate(X_test_lt5, y_test_lt5, verbose=0)
print("測試資料集的準確度 = {:.2f}".format(accuracy))
print("--------------------------")
# 顯示各神經層
print(len(model.layers))
for i in range(len(model.layers)):
    print(i, model.layers[i])
# 凍結上層模型
for i in range(4):
    model.layers[i].trainable = False
print("--------------------------")
# 編譯模型
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])    
# 訓練模型
history = model.fit(X_train_gte5, y_train_gte5, validation_split=0.2,
                    epochs=5, batch_size=128, verbose=2)
print("--------------------------")
# 評估模型
loss, accuracy = model.evaluate(X_test_gte5, y_test_gte5, verbose=0)
print("測試資料集的準確度 = {:.2f}".format(accuracy))

