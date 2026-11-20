import numpy as np
from keras.datasets import reuters
from keras.utils import pad_sequences
from keras.utils import to_categorical

np.random.seed(10)  # 指定亂數種子
# 載入Reuters資料集
top_words = 10000
(X_train, y_train), (X_test, y_test) = reuters.load_data(
                                       num_words=top_words)
# 資料預處理
max_words = 200 
X_train = pad_sequences(X_train, maxlen=max_words)
X_test = pad_sequences(X_test, maxlen=max_words)

print("X_train.shape: ", X_train.shape)
print("X_test.shape: ", X_test.shape)
# One-hot編碼
y_train = to_categorical(y_train, 46)
y_test = to_categorical(y_test, 46)
 
