import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

# 指定亂數種子
np.random.seed(7)
# 載入資料集
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# 將圖片轉換成 4D 張量
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype("float32")
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype("float32")
print("X_train Shape: ", X_train.shape)
print("X_test Shape: ", X_test.shape)
print("--------------------------")
# 因為是固定範圍, 所以執行正規化, 從 0-255 至 0-1
X_train = X_train / 255
X_test = X_test / 255
# One-hot編碼
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print("y_train Shape: ", y_train.shape)
print(y_train[0])
