import numpy as np
from keras.utils import text_dataset_from_directory
from keras.layers import TextVectorization
from keras import Sequential
from keras.layers import Input, Dense, Dropout, Embedding, GlobalMaxPooling1D
import tensorflow as tf
import string, re

np.random.seed(10)  # 指定亂數種子
# 建立訓練,驗證和測試資料集
batch_size = 32
raw_train_ds, raw_val_ds = text_dataset_from_directory(
    "/Dataset/aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="both",
    seed=1337
)
raw_test_ds = text_dataset_from_directory(
    "/Dataset/aclImdb/test", 
    batch_size=batch_size,
    seed=1337
)
print("--------------------------")
# 轉換成小寫, 刪除(<br>標籤和標點符號
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )
# 參數設定
top_words = 1000
max_words = 100

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=top_words,
    output_mode="int",
    output_sequence_length=max_words,
)
# 刪除 labels 標籤
text_ds = raw_train_ds.map(lambda x, y: x)
# 呼叫 adapt()函式建立字彙表
vectorize_layer.adapt(text_ds)
# 建立 vectorize_text函式執行文字資料向量化
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label
# 呼叫 vectorize_text()函式執行文字資料向量化
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)
# 定義模型
model = Sequential()
model.add(Input(shape=(max_words, ), dtype="int64"))
model.add(Embedding(top_words, 32))
model.add(Dropout(0.25))
model.add(GlobalMaxPooling1D())  # 不可用Flatten層
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(1, activation="sigmoid"))
model.summary()   # 顯示模型摘要資訊
print("--------------------------")
# 編譯模型
model.compile(loss="binary_crossentropy", optimizer="adam", 
              metrics=["accuracy"])
# 訓練模型
history = model.fit(train_ds, validation_data=val_ds, 
          epochs=5, verbose=2)
print("--------------------------")
# 評估模型
loss, accuracy = model.evaluate(test_ds, verbose=0)
print("測試資料集的準確度 = {:.2f}".format(accuracy))        

