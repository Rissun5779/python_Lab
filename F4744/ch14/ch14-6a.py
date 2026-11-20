import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils import to_categorical

np.random.seed(10)  # 指定亂數種子
# 載入糖尿病資料集
df = pd.read_csv("./diabetes.csv")
dataset = df.values
np.random.shuffle(dataset)  # 使用亂數打亂資料
# 分割成特徵資料和標籤資料
X = dataset[:, 0:8]
y = dataset[:, 8]
# 特徵標準化
X -= X.mean(axis=0)
X /= X.std(axis=0)
# One-hot編碼
y = to_categorical(y)
# 分割訓練和測試資料集
X_train, y_train = X[:690], y[:690]     # 訓練資料前690筆
X_test, y_test = X[690:], y[690:]       # 測試資料後78筆
# 建立Keras的Sequential模型
model = load_model("best_model.keras")
# 測試資料集的預測值
predict_values = model.predict(X_test, batch_size=10, verbose=0)
print(predict_values[0])

