import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 
np.random.seed(7)  # 指定亂數種子
# 載入加州房屋資料集
df = pd.read_csv("./california_housing.csv")
dataset = df.values
np.random.shuffle(dataset)  # 使用亂數打亂資料
# 分割成特徵資料和標籤資料
X = dataset[:, 0:8]
y = dataset[:, 8]
print(X.shape, y.shape)
print("--------------------------")
# 分割成訓練和測試資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                   random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print("--------------------------")
# 執行特徵標準化
sc = StandardScaler()
X_train = sc.fit_transform(X_train.astype("float"))
X_test = sc.transform(X_test.astype("float"))
# 載入Keras模型
model = load_model("california_housing.keras")
# 預測房屋價值
print("\nPredicting ...")
price = model.predict(X_test[1:2], verbose=0)
print("測試資料的房屋價值:", y_test[1])
print("模型預測的房屋價值:", price[0][0])




