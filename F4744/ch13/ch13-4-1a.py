from keras.utils import load_img, img_to_array
# 載入圖檔
img = load_img("penguins.png")
# 將圖片轉換成NumPy陣列
img_array = img_to_array(img)
print(img_array.dtype)
print(img_array.shape)
