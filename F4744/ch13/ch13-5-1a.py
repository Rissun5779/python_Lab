from keras.utils import load_img
from keras.layers import RandomTranslation
import matplotlib.pyplot as plt
# 載入圖檔
image = load_img("koala.png")

translation = RandomTranslation(height_factor=0.2,
                                width_factor=0.2) 
output = translation(image).numpy().astype("uint8")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 5))
ax1.imshow(image)
ax1.set_title("original")
ax1.set_axis_off()
# translation
ax2.imshow(output)
ax2.set_title("translation")
ax2.set_axis_off()
plt.show()