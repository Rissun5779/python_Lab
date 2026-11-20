from keras.datasets import reuters

# 載入 Reuters 資料集, 如果是第一次載入會自行下載資料集
top_words = 10000
(X_train, y_train), (X_test, y_test) = reuters.load_data(
                                       num_words=top_words)
# 形狀
print("X_train.shape: ", X_train.shape)
print("y_train.shape: ", y_train.shape)
print("X_test.shape: ", X_test.shape)
print("y_test.shape: ", y_test.shape)
print("--------------------------")
# 顯示 Numpy 陣列內容
print(X_train[0])
print(y_train[0])   # 標籤資料
