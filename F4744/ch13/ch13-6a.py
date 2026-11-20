import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import Rescaling, RandomFlip, RandomRotation, RandomZoom
from keras.utils import to_categorical

np.random.seed(10)  # 指定亂數種子
# 載入資料集
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# One-hot編碼
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# 建立圖片增強層的 Sequential模型
data_augmentation = Sequential(
    [
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.1)
    ]
)
# 建立 tf.data.Dataset 套用圖片增強層
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.batch(128).map(lambda x, y: (data_augmentation(x), y))
# 定義模型
model = Sequential()
model.add(Input(shape=X_train.shape[1:]))
model.add(Rescaling(scale=1./255))      # 加上圖片預處理層
model.add(Conv2D(32, kernel_size=(3, 3), padding="same",
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), padding="same",
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))
model.summary()   # 顯示模型摘要資訊
print("--------------------------")
# 編譯模型
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])
# 訓練模型
model.fit(train_ds, epochs=100, verbose=2,
          validation_data=(X_test, y_test))
# 評估模型
print("\nTesting ...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("測試資料集的準確度 = {:.2f}".format(accuracy))
