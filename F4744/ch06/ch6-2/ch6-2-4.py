import numpy as np
import pandas as pd
from keras.models import load_model

np.random.seed(7)
# 載入Titanic的測試資料集
df_test = pd.read_csv("./titanic_test.csv")
dataset_test = df_test.values.astype('float32')
# 分割成特徵資料和標籤資料
X_test = dataset_test[:, 0:9]
y_test = dataset_test[:, 9]
# 特徵標準化
X_test -= X_test.mean(axis=0)
X_test /= X_test.std(axis=0)
# 載入Keras模型
model = load_model("titanic.keras")
# 計算分類的預測值
print("\nPredicting ...")
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)
y_pred = y_pred.flatten()
print(y_pred)
print(y_test.astype(int))
print("--------------------------")
# 顯示混淆矩陣
tb = pd.crosstab(y_test.astype(int), y_pred,
                 rownames=["label"], colnames=["predict"])
print(tb)
tb.to_html("ch6-2-4.html")

