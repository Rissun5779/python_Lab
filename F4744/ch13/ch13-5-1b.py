from keras.utils import load_img
from keras.layers import RandomCrop
import matplotlib.pyplot as plt
# 載入圖檔
image = load_img("koala.png")

height, width = 200, 200
crop = RandomCrop(height, width)
output = crop(image).numpy().astype("uint8")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 5))
ax1.imshow(image)
ax1.set_title("original")
ax1.set_axis_off()
# crop
ax2.imshow(output)
ax2.set_title("crop")
ax2.set_axis_off()
plt.show()