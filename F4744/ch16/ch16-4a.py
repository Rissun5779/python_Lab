import numpy as np
from keras.datasets import mnist
from keras.layers import Layer
from keras import Model, Sequential, activations, ops, random
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical

# 指定亂數種子
np.random.seed(7)
# 載入資料集
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# 因為是固定範圍, 所以執行正規化, 從 0-255 至 0-1
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255
# 轉換形狀成為 (28, 28, 1)
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)
# One-hot編碼
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# 客製化Dense層
class MyDense(Layer):
    def __init__(self, units, activation=None, name=None):
        super().__init__(name=name)
        self.units = units
        self.activation = activations.get(activation)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.w = self.add_weight(
            shape=(input_dim, self.units),
            initializer="glorot_normal",
            name="kernel",
            trainable=True,
        )

        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            name="bias",
            trainable=True,
        )

    def call(self, inputs):
        x = ops.matmul(inputs, self.w) + self.b
        return self.activation(x)
# 客製化Dropout層
class MyDropout(Layer):
    def __init__(self, rate, name=None):
        super().__init__(name=name)
        self.rate = rate
        self.seed_generator = random.SeedGenerator(1337)

    def call(self, inputs):
        return random.dropout(inputs, self.rate, 
                              seed=self.seed_generator)
# 客製化模型
class MyModel(Model):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn_base = Sequential(
            [
                Input(shape=(28, 28, 1)),
                Conv2D(16, kernel_size=(5, 5), padding="same",
                       activation="relu"),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(32, kernel_size=(5, 5), padding="same",
                       activation="relu"),
                MaxPooling2D(pool_size=(2, 2)),
                MyDropout(0.5),
                Flatten(),
                MyDense(128, activation="relu")
            ]
        )
        self.dp = MyDropout(0.5)
        self.dense = MyDense(num_classes, activation="softmax")

    def call(self, x):
        x = self.cnn_base(x)
        x = self.dp(x)
        return self.dense(x)
# 定義模型
model = MyModel(num_classes=10)
model.summary()
print("--------------------------")
# 編譯模型
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])
# 訓練模型
history = model.fit(X_train, y_train, validation_split=0.2,
                    epochs=10, batch_size=128, verbose=2)
# 評估模型
print("\nTesting ...")
loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
print("訓練資料集的準確度 = {:.2f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("測試資料集的準確度 = {:.2f}".format(accuracy))


