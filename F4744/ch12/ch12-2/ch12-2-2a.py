import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

np.random.seed(10)  # 指定亂數種子
look_back = 60
print("回看天數:", look_back)

def create_dataset(ds, look_back=1):
    X_data, y_data = [],[]
    for i in range(len(ds)-look_back):
        X_data.append(ds[i:(i+look_back), 0])
        y_data.append(ds[i+look_back, 0])
    
    return np.array(X_data), np.array(y_data)

# 使用模型預測股價 - 2017年2~3月預測 4 月份股價
df_test = pd.read_csv("GOOG_Stock_Price_Test.csv")
X_test_set = df_test.iloc[:,4:5].values
# 產生特徵資料和標籤資料
X_test, y_test = create_dataset(X_test_set, look_back)
# 特徵標準化 - 正規化
sc = MinMaxScaler() 
X_test = sc.fit_transform(X_test)
y_test_scaler = sc.fit_transform(y_test.reshape(-1, 1))
# 轉換成(樣本數, 時步, 特徵)張量
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# 載入Keras的Sequential模型
model = load_model("stockprice"+str(look_back)+".keras")
# 預測模型
y_test_pred = model.predict(X_test)
# 將預測值轉換回股價
y_test_pred_price = sc.inverse_transform(y_test_pred)
# 繪出股價圖表
import matplotlib.pyplot as plt

plt.plot(y_test, color="red", label="Real Stock Price")
plt.plot(y_test_pred_price, color="blue", label="Predicted Stock Price")
plt.title("2017 Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Google Time Price")
plt.legend()
plt.show()

