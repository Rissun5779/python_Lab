from keras.models import load_model

# 建立Keras的Sequential模型
model = load_model("imdb_rnn.keras")
model.summary()   # 顯示模型摘要資訊
# 顯示SimpleRNN層的權重形狀
print(2, model.layers[2].name, ":")
weights = model.layers[2].get_weights()
for i in range(len(weights)):
    print("==>", i, weights[i].shape)