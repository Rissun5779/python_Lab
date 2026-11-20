import numpy as np
from keras.datasets import imdb
from keras.utils import pad_sequences
from keras import Sequential
from keras.layers import Input, Dense, Embedding, Flatten

np.random.seed(10)  # 指定亂數種子
# 載入 IMDb 資料集
top_words = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(
                                num_words=top_words)
# 資料預處理
max_words = 500
X_train = pad_sequences(X_train, maxlen=max_words)
X_test = pad_sequences(X_test, maxlen=max_words)
# 定義模型
model = Sequential()
model.add(Input(shape=(max_words, )))
model.add(Embedding(top_words, 32))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()   # 顯示模型摘要資訊
print("--------------------------")
# 編譯模型
model.compile(loss="binary_crossentropy", optimizer="adam", 
              metrics=["accuracy"])
# 訓練模型
history = model.fit(X_train, y_train, validation_split=0.2, 
          epochs=5, batch_size=128, verbose=2)
print("--------------------------")
# 評估模型
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("測試資料集的準確度 = {:.2f}".format(accuracy))

