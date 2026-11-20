import pandas as pd
import numpy as np
from keras import Sequential 
from keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 
np.random.seed(7)  # 指定亂數種子
# 載入加州房屋資料集
df = pd.read_csv("./california_housing.csv")
dataset = df.values
np.random.shuffle(dataset)  # 使用亂數打亂資料
# 分割成特徵資料和標籤資料
X = dataset[:, 0:8]
y = dataset[:, 8]
print(X.shape, y.shape)
print("--------------------------")
# 分割成訓練和測試資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                   random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print("--------------------------")
# 執行特徵標準化
sc = StandardScaler()
X_train = sc.fit_transform(X_train.astype("float"))
X_test = sc.transform(X_test.astype("float"))
# 建立Keras的Sequential模型
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(16, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(1))
model.summary()
print("--------------------------")
# 編譯模型
model.compile(optimizer='adam', loss='mse', metrics=["root_mean_squared_error"])
# 訓練模型
history = model.fit(X_train, y_train, batch_size=128, 
                    validation_split=0.2, epochs=100, verbose=2)
print("--------------------------")
# 評估模型
loss, rmse = model.evaluate(X_test, y_test, verbose=0)
print("測試資料集的RMSE = {:.2f}".format(rmse))
# 儲存Keras模型
print("Saving Model: california_housing.keras ...")
model.save("california_housing.keras")
# 顯示訓練和驗證損失圖表
import matplotlib.pyplot as plt

loss = history.history["loss"]
epochs = range(1, len(loss)+1)
val_loss = history.history["val_loss"]
plt.plot(epochs, loss, "bo", label="Training Loss")
plt.plot(epochs, val_loss, "r", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
# 顯示訓練和驗證RMSE
rmse = history.history["root_mean_squared_error"]
epochs = range(1, len(rmse)+1)
val_rmse = history.history["val_root_mean_squared_error"]
plt.plot(epochs, rmse, "b-", label="Training RMSE")
plt.plot(epochs, val_rmse, "r--", label="Validation RMSE")
plt.title("Training and Validation RMSE")
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.legend()
plt.show()

