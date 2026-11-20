import numpy as np
from keras.datasets import cifar10
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras import Sequential
from keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from keras.utils import to_categorical
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")
# 指定亂數種子
np.random.seed(10)
# 載入資料集
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# 打亂 2 個 Numpy 陣列
def randomize(a, b):
    permutation = list(np.random.permutation(a.shape[0]))
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    
    return shuffled_a, shuffled_b

X_train, y_train = randomize(X_train, y_train)
X_test, y_test = randomize(X_test, y_test)
# 取出部分的訓練和測試資料
X_train = X_train[:5000]
y_train = y_train[:5000]
X_test = X_test[:1000]
y_test = y_test[:1000]
# One-hot編碼
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
# 批次載入資料, 並且執行資料預處理和調整圖片尺寸
batch_size = 32

def preprocess_data(images, labels):
    # 調整圖片尺寸
    images = tf.image.resize(images, (200, 200))
    images = tf.cast(images, tf.float32)
    # 資料前處理 
    images = preprocess_input(images)    
    return images, labels

train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .batch(batch_size)
    .map(lambda x, y: preprocess_data(x, y))
    .prefetch(tf.data.AUTOTUNE)
)
test_dataset = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .batch(batch_size)
    .map(lambda x, y: preprocess_data(x, y))
    .prefetch(tf.data.AUTOTUNE)
)
# 載入 ResNet50 模型
resnet_model = ResNet50(weights="imagenet",
                        include_top=False,
                        input_shape=(200, 200, 3))
# 使用 ResNet50 模型預測訓練資料的特徵資料
print("使用 ResNet50 模型預測訓練資料的特徵資料...")
train_features = resnet_model.predict(train_dataset)
print("使用 ResNet50 模型預測測試資料的特徵資料...")
test_features = resnet_model.predict(test_dataset)
# 定義模型
model = Sequential()
model.add(Input(shape=train_features.shape[1:]))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))
# 編譯模型
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])
# 訓練模型
history = model.fit(train_features, y_train,
                    validation_data=(test_features, y_test), 
                    epochs=14, batch_size=batch_size, verbose=2)
# 評估模型
print("\nTesting ...")
loss, accuracy = model.evaluate(test_features, y_test, verbose=0)
print("測試資料集的準確度 = {:.2f}".format(accuracy))
# 顯示圖表來分析模型的訓練過程
import matplotlib.pyplot as plt
# 顯示訓練和驗證損失
loss = history.history["loss"]
epochs = range(1, len(loss)+1)
val_loss = history.history["val_loss"]
plt.plot(epochs, loss, "bo-", label="Training Loss")
plt.plot(epochs, val_loss, "ro--", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
# 顯示訓練和驗證準確度
acc = history.history["accuracy"]
epochs = range(1, len(acc)+1)
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "bo-", label="Training Acc")
plt.plot(epochs, val_acc, "ro--", label="Validation Acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
