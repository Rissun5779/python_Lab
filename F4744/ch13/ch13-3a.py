from keras.utils import text_dataset_from_directory
from keras.layers import TextVectorization
import tensorflow as tf
import string, re

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
# 刪除標籤和轉換成小寫
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )
# 參數設定
top_words = 1000
max_words = 100
# 使用TextVectorization文字向量層執行文字預處理
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
imdb_vocab = vectorize_layer.get_vocabulary()
# 建立詞索引字典
imdb_index_lookup = dict(zip(range(len(imdb_vocab)), imdb_vocab))
for idx in range(10):
    print(idx, ":", imdb_index_lookup[idx], end=", ")
print()
print("--------------------------")
# 建立 vectorize_text函式執行文字資料向量化
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label
# 呼叫 vectorize_text()函式執行文字資料向量化
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

for text_batch, label_batch in train_ds.take(1):
    for i in range(5):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])
        print("--------------------------")
        
        



