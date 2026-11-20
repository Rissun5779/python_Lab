# Transformer模型的IMDb情感分析
import numpy as np
from keras.datasets import imdb
from keras.utils import pad_sequences
from keras import Model, Sequential
from keras.layers import Layer, MultiHeadAttention, Input, Dense, Dropout
from keras.layers import LayerNormalization, Embedding, GlobalAveragePooling1D
import warnings

warnings.filterwarnings("ignore")  
np.random.seed(7)   # 指定亂數種子
# 實作Transformer神經層區塊
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads,
                                      key_dim=embed_dim)
        self.ffn = Sequential(
            [
                Dense(ff_dim, activation="relu"),
                Dense(embed_dim)
            ]
        )
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()

    def call(self, inputs):
        attn_output = self.att(query=inputs, value=inputs,
                               key=inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

# 實作位置編碼與詞嵌入層
class TokenAndPositionEmbedding(Layer):
    def __init__(self, max_words, top_words, embed_dim):
        super().__init__()
        self.token_emb = Embedding(input_dim=top_words,
                                   output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=max_words,
                                 output_dim=embed_dim)

    def call(self, x):
        maxlen = x.shape[-1]
        positions = np.arange(start=0, stop=maxlen, step=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

# 載入 IMDb 資料集, 如果是第一次載入會自行下載資料集
top_words = 20000
(X_train, y_train), (X_test, y_test) = imdb.load_data(
                                num_words=top_words)
# 形狀
print("X_train: (",len(X_train),",",len(X_train[0]), ")")
print("y_train: (",len(y_train),",)")
print("X_test: (",len(X_test),",",len(X_test[0]), ")")
print("y_test: (",len(y_test),",)")
print("--------------------------")
# 資料預處理
max_words = 100
X_train = pad_sequences(X_train, maxlen=max_words)
X_test = pad_sequences(X_test, maxlen=max_words)
# 定義模型參數
embed_dim = 32  # 每一個單字的嵌入維度尺寸
num_heads = 2   # 多頭注意力機制中的頭數
ff_dim = 32     # 轉換器內部前馈神經網絡的隱藏層尺寸
# 定義模型
inputs = Input(shape=(max_words,))
embedding_layer = TokenAndPositionEmbedding(max_words, 
                                            top_words,
                                            embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads,
                                     ff_dim)
x = transformer_block(x)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.1)(x)
x = Dense(20, activation="relu")(x)
x = Dropout(0.1)(x)
outputs = Dense(2, activation="softmax")(x)
model = Model(inputs=inputs, outputs=outputs)
model.summary()   # 顯示模型摘要資訊
print("--------------------------")
# 編譯模型
model.compile(optimizer="adam", 
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
# 訓練模型
epochs = 5 
history = model.fit(X_train, y_train, batch_size=32, 
                    epochs=epochs, validation_split=0.2)
print("--------------------------")
# 評估模型
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("測試資料集的準確度 = {:.2f}".format(accuracy))
print("--------------------------")
word_index = imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3
word_map = dict([(value, key) for (key, value)
                                  in word_index.items()])
class_names = ["negative", "postive"]
scores = model.predict(X_test[:1])
print(" ".join(word_map.get(w) for w in X_test[0:1].flatten() if w != 0))
print(class_names[np.argmax(scores)], "-", scores)
scores = model.predict(X_test[1:2])
print(" ".join(word_map.get(w) for w in X_test[1:2].flatten() if w != 0))
print(class_names[np.argmax(scores)], "-", scores)

