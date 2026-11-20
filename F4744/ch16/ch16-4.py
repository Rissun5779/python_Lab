import numpy as np
import pandas as pd
from keras import Model
from keras.layers import Layer, Input, Activation
from keras import ops, activations

np.random.seed(10)  # 指定亂數種子
# 載入糖尿病資料集
df = pd.read_csv("./diabetes.csv")
dataset = df.values
np.random.shuffle(dataset)  # 使用亂數打亂資料
# 分割成特徵資料和標籤資料
X = dataset[:, 0:8]
y = dataset[:, 8]
# 客製化Dense層
class MyDense(Layer):
    def __init__(self, units, name=None):
        super().__init__(name=name)
        self.units = units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.w = self.add_weight(
            shape=(input_dim, self.units),
            initializer= "glorot_normal",
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
        return x
# 客製化MLP區塊
class MLPBlock(Layer):
    def __init__(self):
        super().__init__()
        self.dense_1 = MyDense(10)
        self.dense_2 = MyDense(8)
        self.dense_3 = MyDense(1)

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = activations.relu(x)
        x = self.dense_2(x)
        x = activations.relu(x)
        return self.dense_3(x)    
# 定義模型
data_input = Input(shape=(8,))
mlp = MLPBlock()
hidden = mlp(data_input)
output = Activation("sigmoid")(hidden)
model = Model(inputs=data_input, outputs=output)
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

