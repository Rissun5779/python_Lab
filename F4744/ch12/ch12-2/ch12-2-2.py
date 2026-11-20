import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import Input, Dense, Dropout, LSTM

np.random.seed(10)  # 指定亂數種子
# 載入Google股價的訓練資料集
df_train = pd.read_csv("GOOG_Stock_Price_Train.csv",
                       index_col="Date", parse_dates=True)
X_train_set = df_train.iloc[:,4:5].values  # Adj Close欄位
# 取出幾天前股價來建立成特徵和標籤資料集
def create_dataset(ds, look_back=1):
    X_data, y_data = [],[]
    for i in range(len(ds)-look_back):
        X_data.append(ds[i:(i+look_back), 0])
        y_data.append(ds[i+look_back, 0])
    
    return np.array(X_data), np.array(y_data)

look_back = 60
print("回看天數:", look_back)
print("--------------------------")
# 分割成特徵資料和標籤資料
X_train, y_train = create_dataset(X_train_set, look_back)
# 特徵標準化 - 正規化
sc = MinMaxScaler() 
X_train = sc.fit_transform(X_train)
y_train = sc.fit_transform(y_train.reshape(-1, 1))
# 轉換成(樣本數, 時步, 特徵)張量
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print("X_train.shape: ", X_train.shape)
print("y_train.shape: ", y_train.shape)
# 定義模型
model = Sequential()
model.add(Input(shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=True,
               activation="tanh"))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True,
               activation="tanh"))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False,
               activation="tanh"))
model.add(Dropout(0.2))
model.add(Dense(1))
model.summary()   # 顯示模型摘要資訊
print("--------------------------")
# 編譯模型
model.compile(loss="mse", optimizer="adam")
# 訓練模型
model.fit(X_train, y_train, epochs=100, batch_size=32)
print("--------------------------")
# 儲存Keras模型
print("Saving Model: stockprice"+str(look_back)+".keras ...")
model.save("stockprice"+str(look_back)+".keras")

