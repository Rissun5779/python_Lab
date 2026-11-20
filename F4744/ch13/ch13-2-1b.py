from keras.models import load_model

# 建立Keras的Sequential模型
model = load_model("mnist.keras")
model.summary()   # 顯示模型摘要資訊
# 顯示各層的權重形狀
for i in range(len(model.layers)):
    print(i, model.layers[i].name, ":")
    weights = model.layers[i].get_weights()
    for j in range(len(weights)):
        print("==>", j, weights[j].shape)