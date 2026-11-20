import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Input, Dense

np.random.seed(10)  # 指定亂數種子
# 建立測試資料
X = np.linspace(-1, 1, 300)
np.random.shuffle(X)  # 使用亂數打亂資料
y = 0.3 * X + 5 + np.random.normal(0, 0.05, (300, ))
# 繪出資料的散佈圖
plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("y")

plt.show()

X_train, y_train = X[:270], y[:270]     # 訓練資料前270點
X_test, y_test = X[270:], y[270:]       # 測試資料後30點

# 建立Keras的Sequential模型
model = Sequential()
model.add(Input(shape=(1,)))
model.add(Dense(2))                     # 隱藏層 2個神經元
model.add(Dense(1))                     # 輸出層 1個神經元
# 編譯模型
model.compile(loss="mse", optimizer="sgd", metrics=["accuracy"])
model.summary()

# 訓練模型
print("Training ....")
history = model.fit(X_train, y_train, validation_split=0.2,
                    epochs=10, batch_size=30)

# 顯示訓練和驗證損失
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, "bo", label="Training Loss")
plt.plot(epochs, val_loss, "b", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()

# 測試模型
print("\nTesting ...")
cost = model.evaluate(X_test, y_test, batch_size=30)
print("Test Cost =", cost)
print("\nHidden Layer ...")
W, b = model.layers[0].get_weights()
print("Weights=", W, "\nbiases=", b)  
print("\nOutput Layer ...")
W, b = model.layers[1].get_weights()
print("Weights=", W, "\nbiases=", b)

# 預測模型
y_pred = model.predict(X_test)
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.xlabel("X")
plt.ylabel("y")

plt.show()
