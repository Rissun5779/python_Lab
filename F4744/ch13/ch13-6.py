import numpy as np
from keras.utils import image_dataset_from_directory
from keras import Sequential
from keras.layers import RandomFlip, RandomRotation, RandomZoom, Rescaling
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.losses import SparseCategoricalCrossentropy

np.random.seed(10)  # 指定亂數種子
# 載入資料集
train_ds, val_ds = image_dataset_from_directory("/Dataset/cats_dogs_light/train",
                          validation_split=0.2, subset="both",
                          image_size=(180,180), interpolation="bilinear",
                          crop_to_aspect_ratio=True,
                          seed=42, shuffle=True, batch_size=32)

test_ds = image_dataset_from_directory("/Dataset/cats_dogs_light/test",
                            image_size=(180,180), interpolation="bilinear",
                            crop_to_aspect_ratio=True,
                            seed=42, shuffle=True, batch_size=32)

normalization_layer = Rescaling(1./255)
# 使用圖片預處理層
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
# 定義模型
model = Sequential([
    Input(shape=(180,180, 3)), 
    RandomFlip("horizontal"),      
    RandomRotation(0.2),
    RandomZoom(0.2),  
    Conv2D(32, 3, activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2)
])
model.summary()   # 顯示模型摘要資訊
print("--------------------------")
# 編譯模型
model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# 訓練模型  
history = model.fit(train_ds, epochs=40, 
                    validation_data=val_ds, verbose=2)
print("--------------------------")
# 評估模型
loss, accuracy = model.evaluate(test_ds, verbose=0)
print("測試資料集的準確度 = {:.2f}".format(accuracy))        


