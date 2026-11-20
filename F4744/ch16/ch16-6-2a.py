# 英文翻譯成中文的Transformer翻譯模型
import numpy as np
import random, re
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings
from keras.layers import Layer, MultiHeadAttention, Input, Dense
from keras.layers import LayerNormalization, Embedding, Dropout
from keras.layers import TextVectorization
from keras import Sequential, Model
from keras import ops
from opencc import OpenCC
import jieba

np.random.seed(7)   # 指定亂數種子
# 中文語料檔案http://www.manythings.org/anki/cmn-eng.zip
text_file = "cmn.txt"  
# 剖析語料資料成為(eng, cht)元組的串列, 和在cht前後加上"[start]"和"[end]"
with open(text_file, encoding='utf-8') as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []
cc = OpenCC("s2tw.json")
for line in lines:
    line, _ = line.split("CC-BY")  # 分割字串
    line = line.strip()            # 刪除空白字元
    eng, cht = line.split("\t")
    # 轉換成繁體中文和斷詞
    cht = jieba.lcut(cc.convert(cht)) 
    cht = ' '.join(cht)
    cht = "[start] " + cht + " [end]"
    text_pairs.append((eng, cht))
print("--------------------------")    
# 隨機顯示1筆文字對的訓練資料
print(random.choice(text_pairs))
print("--------------------------")
# 分割成訓練, 驗證和測試資料集
np.random.shuffle(text_pairs)  # 打亂資料
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]
print("資料對數量:", len(text_pairs))
print("訓練資料集數量:", len(train_pairs))
print("驗證資料集數量: ", len(val_pairs))
print("測試資料集數量:", len(test_pairs))
print("--------------------------")
# 資料預處理: 文字資料向量化, 首先是參數設定
top_words = 10000
max_words = 50
batch_size = 64
# 刪除strip_chars變數的字元
def cht_custom_standardization(input_string):
    strip_chars = """！？。｡＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"""
    return tf_strings. \
           regex_replace(input_string,"[%s]" % re.escape(strip_chars),"")

# 使用TextVectorization文字向量層執行文字預處理
eng_vectorization = TextVectorization(
    max_tokens=top_words,
    output_mode="int",
    output_sequence_length=max_words,
)
cht_vectorization = TextVectorization(
    max_tokens=top_words,
    output_mode="int",
    output_sequence_length=max_words + 1,
    standardize=cht_custom_standardization,
)
# 從文字對分別取出訓練資料的英文和中文資料
train_eng_texts = [pair[0] for pair in train_pairs]
train_cht_texts = [pair[1] for pair in train_pairs]
# 呼叫 adapt()函式建立字彙表
eng_vectorization.adapt(train_eng_texts)
cht_vectorization.adapt(train_cht_texts)
# 使用TensorFlow的Dataset物件來格式化資料集,和批次載入資料
# 格式化資料集
def format_dataset(eng, cht):
    eng = eng_vectorization(eng)
    cht = cht_vectorization(cht)
    return (
        {
            "encoder_inputs": eng,
            "decoder_inputs": cht[:, :-1],
        },
        cht[:, 1:],
    )
# 建立資料集
def make_dataset(pairs):
    eng_texts, cht_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    cht_texts = list(cht_texts)
    dataset = tf_data.Dataset.from_tensor_slices((eng_texts, cht_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.cache().shuffle(2048).prefetch(16)

# 建立訓練和驗證資料集
train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)
# 顯示一個批次的資料集形狀
for inputs, targets in train_ds.take(1):
    print("inputs['encoder_inputs'].shape:", inputs["encoder_inputs"].shape)
    print("inputs['decoder_inputs'].shape:", inputs["decoder_inputs"].shape)
    print("targets.shape:", targets.shape)
print("--------------------------")
# 實作Transformer編碼器
class TransformerEncoder(Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(num_heads=num_heads,
                                            key_dim=embed_dim)
        self.dense_proj = Sequential(
            [
                Dense(dense_dim, activation="relu"),
                Dense(embed_dim),
            ]
        )
        self.layernorm_1 = LayerNormalization()
        self.layernorm_2 = LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = ops.cast(mask[:, None, :], dtype="int32")
        else:
            padding_mask = None

        attention_output = self.attention(query=inputs, value=inputs,
                                          key=inputs,
                                          attention_mask=padding_mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "dense_dim": self.dense_dim,
                "num_heads": self.num_heads,
            }
        )
        return config

# 實作位置編碼與詞嵌入層
class PositionalEmbedding(Layer):
    def __init__(self, max_words, top_words, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = Embedding(input_dim=top_words,
                                          output_dim=embed_dim)
        self.position_embeddings = Embedding(input_dim=max_words,
                                             output_dim=embed_dim)
        self.sequence_length = max_words
        self.vocab_size = top_words
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = ops.shape(inputs)[-1]
        positions = ops.arange(0, length, 1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        else:
            return ops.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config

# 實作Transformer解碼器
class TransformerDecoder(Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = MultiHeadAttention(num_heads=num_heads,
                                              key_dim=embed_dim)
        self.attention_2 = MultiHeadAttention(num_heads=num_heads,
                                              key_dim=embed_dim)
        self.dense_proj = Sequential(
            [
                Dense(latent_dim, activation="relu"),
                Dense(embed_dim),
            ]
        )
        self.layernorm_1 = LayerNormalization()
        self.layernorm_2 = LayerNormalization()
        self.layernorm_3 = LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = ops.cast(mask[:, None, :], dtype="int32")
            padding_mask = ops.minimum(padding_mask, causal_mask)
        else:
            padding_mask = None

        attention_output_1 = self.attention_1(query=inputs, value=inputs,
                                              key=inputs,
                                              attention_mask=causal_mask)
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = ops.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = ops.arange(sequence_length)[:, None]
        j = ops.arange(sequence_length)
        mask = ops.cast(i >= j, dtype="int32")
        mask = ops.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = ops.concatenate(
            [ops.expand_dims(batch_size, -1), ops.convert_to_tensor([1, 1])],
            axis=0,
        )
        return ops.tile(mask, mult)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "latent_dim": self.latent_dim,
                "num_heads": self.num_heads,
            }
        )
        return config

# 定義Transformer模型的參數
embed_dim = 256
latent_dim = 2048
num_heads = 8
# 定義Transformer模型
encoder_inputs = Input(shape=(None,), dtype="int64",
                             name="encoder_inputs")
x = PositionalEmbedding(max_words, top_words, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
encoder = Model(encoder_inputs, encoder_outputs)

decoder_inputs = Input(shape=(None,), dtype="int64",
                       name="decoder_inputs")
encoded_seq_inputs = Input(shape=(None, embed_dim), 
                           name="decoder_state_inputs")
x = PositionalEmbedding(max_words, top_words, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs)
x = Dropout(0.5)(x)
decoder_outputs = Dense(top_words, activation="softmax")(x)
decoder = Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

decoder_outputs = decoder([decoder_inputs, encoder_outputs])
transformer = Model([encoder_inputs, decoder_inputs], 
                    decoder_outputs, name="transformer")
transformer.summary()
print("--------------------------")
# 編譯模型
transformer.compile(optimizer="rmsprop", 
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])
# 訓練模型
epochs = 5  # 因為只是展示, 所以訓練週期只有5次, 最少需40次
transformer.fit(train_ds, epochs=epochs, validation_data=val_ds)
print("--------------------------")
# 建立英文譯成中文的解碼函式, 使用的是Transformer翻譯模型
# 可以從"[start]"開始預測下一個字詞, 直到"[end]"為止
# 取得中文字彙表的字詞索引字典
cht_vocab = cht_vectorization.get_vocabulary()
# 建立中文的詞索引轉換表
cht_index_lookup = dict(zip(range(len(cht_vocab)), cht_vocab))
# 最大解碼的句子長度
max_decoded_sentence_length = 50
# 建立解碼函式
def decode_sequence(input_sentence):
    # 向量化英文的句子
    tokenized_input_sentence = eng_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        # 向量化中文的句子
        tokenized_target_sentence = cht_vectorization(
                                    [decoded_sentence])[:, :-1]
        # 使用Transformer翻譯模型預測下一個字詞
        predictions = transformer([tokenized_input_sentence, 
                                   tokenized_target_sentence])
        # 取得最有可能字詞的詞索引
        sampled_token_index = ops.convert_to_numpy(
            ops.argmax(predictions[0, i, :])
        ).item(0)
        # 從詞索引轉換成中文的字詞
        sampled_token = cht_index_lookup[sampled_token_index]
        # 將預測出的中文字詞加至解碼句子的最後
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
        
    return decoded_sentence

# 取出測試資料集的英文句子
test_eng_texts = [pair[0] for pair in test_pairs]
# 執行30次測試翻譯
for _ in range(30):
    # 隨機選擇句子
    input_sentence = random.choice(test_eng_texts)
    print(">", input_sentence)
    # 呼叫函式翻譯輸入的英文句子成為解碼後的中文句子
    translated = decode_sequence(input_sentence)
    print(">>", translated)

