from keras.utils import img_to_array
from keras.utils import load_img
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

# 建立 VGG16 模型
model = VGG16(weights="imagenet", include_top=True) 
# 載入測試圖片
img = load_img("koala.png", target_size=(224, 224))
X = img_to_array(img)    # 轉換成 Numpy陣列
print("X.shape: ", X.shape)
# Reshape (1, 224, 224, 3)
img = X.reshape((1, X.shape[0], X.shape[1], X.shape[2]))
# 資料預處理
img = preprocess_input(img)
print("img.shape: ", img.shape)
# 使用模型進行預測
y_pred = model.predict(img)
# 解碼預測結果
label = decode_predictions(y_pred)
result = label[0][0]  # 取得最可能的結果
print("%s (%.2f%%)" % (result[1], result[2]*100))
