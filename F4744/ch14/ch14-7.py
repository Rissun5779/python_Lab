import numpy as np
from keras.datasets import fashion_mnist
from keras import Sequential
from keras.layers import Input, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
import keras_tuner as kt

np.random.seed(10)  # 指定亂數種子
# 載入Fashion-MNIST資料集
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
# 正規化
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

def build_model(hp):
   model = Sequential()
   model.add(Input(shape=(28, 28)))
   model.add(Flatten())
   # 最佳化神經元數範圍 32~512
   hp_units = hp.Int("units", min_value=32, max_value=512, step=32)
   model.add(Dense(units=hp_units, activation="relu"))
   if hp.Boolean("dropout"):
       model.add(Dropout(rate=0.25))
   model.add(Dense(10))
   # 最佳化學習率 0.01, 0.001, 或 0.0001
   hp_learning_rate = hp.Choice("learning_rate",
                                values=[1e-2, 1e-3, 1e-4])
   # 編譯模型
   model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                 loss=SparseCategoricalCrossentropy(from_logits=True),
                 metrics=["accuracy"])
   # 回傳模型
   return model
# 建立Hyperband的超參數最佳化
tuner = kt.Hyperband(build_model,
                     objective="val_accuracy",
                     max_epochs=50,
                     factor=3,
                     directory="my_dir",
                     project_name="kt_example")

from keras.callbacks import EarlyStopping

stop_early = EarlyStopping(monitor="val_loss", patience=5)
# 執行超參數最佳化搜尋
tuner.search(X_train, y_train, epochs=50, 
             validation_split=0.2, callbacks=[stop_early])
print("--------------------------")
# 取得最佳的超參數
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Dense層的最佳神經元數:", best_hps.get("units"))
print("Adam最佳的學習率:", best_hps.get("learning_rate"))
print("是否有Dropout層:", best_hps.get("dropout"))
print("--------------------------")
# 建立超參數最佳化模型找出最佳訓練週期, 訓練週期是 50
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=50,
                    validation_split=0.2, verbose=0)
# 找出最佳的訓練週期
val_acc_per_epoch = history.history["val_accuracy"]
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print("最佳的訓練週期:", best_epoch)
print("--------------------------")
# 再次建立超參數最佳化模型
hypermodel = tuner.hypermodel.build(best_hps)
# 使用最佳化週期重新訓練模型
hypermodel.fit(X_train, y_train, epochs=best_epoch, 
               validation_split=0.2, verbose=0)
# 評估最佳化的超參數模型
loss, accuracy = hypermodel.evaluate(X_test, y_test, verbose=0)
print("測試資料集的準確度 = {:.2f}".format(accuracy))

