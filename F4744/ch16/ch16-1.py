from keras import Sequential
from keras.layers import Input, Dense

# 定義模型
model = Sequential()
model.add(Input(shape=(8,)))
model.add(Dense(10, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()  # 顯示模型摘要資訊
# 繪出模型圖片
from keras.utils import plot_model

plot_model(model, to_file="ch16-1.png", show_shapes=True)
