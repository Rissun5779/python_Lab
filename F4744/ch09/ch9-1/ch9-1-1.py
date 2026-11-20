from keras.datasets import cifar10

# 載入 Cifar10 資料集, 如果是第一次載入會自行下載資料集
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# 形狀
print("X_train.shape: ", X_train.shape)
print("y_train.shape: ", y_train.shape)
print("X_test.shape: ", X_test.shape)
print("y_test.shape: ", y_test.shape)
print("--------------------------")
# 顯示 Numpy 二維陣列內容
print(X_train[0])
print(y_train[0])   # 標籤資料

import matplotlib.pyplot as plt

plt.imshow(X_train[0])
plt.title("Label: " + str(y_train[0]))
plt.axis("off")
# 顯示圖片
plt.show()
