import pandas as pd

# 載入加州房屋資料集
df = pd.read_csv("./california_housing.csv")
# 查看前5筆記錄
print(df.head())
df.head().to_html("ch6-3-1a.html")
print("--------------------------")
# 顯示資料集的資訊
print(df.info())

