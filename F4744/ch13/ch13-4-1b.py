from keras.utils import load_img, img_to_array, save_img
# 載入圖檔
img = load_img("penguins.png")
# 將圖片轉換成NumPy陣列
img_array = img_to_array(img)
# 儲存成JPG格式的的圖檔
save_img("penguins.jpg", img_array)

