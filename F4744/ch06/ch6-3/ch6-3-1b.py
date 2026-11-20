import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 載入加州房屋資料集
df = pd.read_csv("./california_housing.csv")
# 使用Pandas顯示視覺化圖表
df.hist(bins=50, figsize=(15,15))
# 使用Seaborn顯示視覺化圖表
plt.figure(figsize = (16,5))
sns.heatmap(df.corr(), annot=True)
