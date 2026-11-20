import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.models import load_model

np.random.seed(7)  # 指定亂數種子
target_mapping = {"setosa": 0,
                  "versicolor": 1,
                  "virginica": 2}
# 載入資料集
df = pd.read_csv("./iris_data.csv")
df["target"] = df["target"].map(target_mapping)
dataset = df.values
np.random.shuffle(dataset)  # 使用亂數打亂資料
# 分割成特徵資料和標籤資料
X = dataset[:,0:4].astype(float)
y = to_categorical(dataset[:,4])
# 特徵標準化
X -= X.mean(axis=0)
X /= X.std(axis=0)
# 分割成訓練和測試資料集
X_train, y_train = X[:120], y[:120]     # 訓練資料前120筆
X_test, y_test = X[120:], y[120:]       # 測試資料後30筆
# 載入Keras模型
model = load_model("iris.keras")
# 計算分類的預測值
print("\nPredicting ...")
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)
y_target = dataset[:,4][120:].astype(int)
print(y_target)
# 顯示混淆矩陣
tb = pd.crosstab(y_target, y_pred, rownames=["label"], 
                 colnames=["predict"])
print(tb)
tb.to_html("ch6-1-3.html")


