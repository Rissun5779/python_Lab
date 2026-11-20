import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import load_model
import warnings

warnings.filterwarnings("ignore")
# 指定亂數種子
np.random.seed(10)
# 載入資料集
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# 因為是固定範圍, 所以執行正規化, 從 0-255 至 0-1
X_test = X_test.astype("float32") / 255
# 載入Keras模型
model = load_model("cifar10.keras")
# 測試資料集的分類和機率的預測值
print("Predicting ...")
y_probs = model.predict(X_test)     # 機率
y_pred = np.argmax(y_probs, axis=1) # 分類
# 建立分類錯誤的 DataFrame 物件
y_test = y_test.flatten()
df = pd.DataFrame({"label":y_test, "predict":y_pred})
df = df[y_test!=y_pred]  # 篩選出分類錯誤的資料
print(df.head())
df.head().to_html("ch9-1-3b.html")
# 隨機選 1 個錯誤分類的數字索引
#np.random.seed(4)
#i = df.sample(n=1).index.values.astype(int)[0]
i = 24
print("Index: ", i)
img = X_test[i] 
# 繪出圖表的預測結果
plt.figure()
plt.subplot(1,2,1)
plt.title("Example of Image:" + str(y_test[i]))
plt.imshow(img, cmap="binary")
plt.axis("off")
plt.subplot(1,2,2)
plt.title("Probabilities for Each Image Class")
plt.bar(np.arange(10), y_probs[i].reshape(10), align="center")
plt.xticks(np.arange(10),np.arange(10).astype(str))
plt.show()
print(((y_probs[i].reshape(10)*100).astype(int)))

