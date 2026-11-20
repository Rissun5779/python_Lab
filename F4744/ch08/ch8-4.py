import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical

# 指定亂數種子
np.random.seed(7)
# 載入資料集
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# 將圖片轉換成 4D 張量
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype("float32")
# 因為是固定範圍, 所以執行正規化, 從 0-255 至 0-1
X_test = X_test / 255
# One-hot編碼
y_test_bk = y_test.copy()   # 備份 y_test 資料集
y_test = to_categorical(y_test)
# 載入Keras模型
model = load_model("mnist.keras")
# 評估模型
print("Testing ...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("測試資料集的準確度 = {:.2f}".format(accuracy))
# 計算分類的預測值
print("\nPredicting ...")
#Y_pred = model.predict_classes(X_test)
predict_x = model.predict(X_test)
y_pred = np.argmax(predict_x , axis=1)
# 顯示混淆矩陣
tb = pd.crosstab(y_test_bk.astype(int), y_pred.astype(int),
                 rownames=["label"], colnames=["predict"])
print(tb)
tb.to_html("ch8-4.html")
