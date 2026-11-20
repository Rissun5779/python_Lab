from keras import Sequential
from keras.layers import Input, Embedding, GRU

# 定義模型
model = Sequential()
model.add(Input(shape=(100, )))
model.add(Embedding(10000, 32))
model.add(GRU(32, return_sequences=True,
               activation="tanh"))
model.add(GRU(32, return_sequences=True,
               activation="tanh"))
model.add(GRU(32, return_sequences=False,
               activation="tanh"))
model.summary()   # 顯示模型摘要資訊
model.compile(loss="binary_crossentropy", optimizer="rmsprop", 
              metrics=["accuracy"])