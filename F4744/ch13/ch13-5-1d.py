from keras.utils import load_img
from keras.layers import RandomFlip
import matplotlib.pyplot as plt
# 載入圖檔
image = load_img("koala.png")

flip = RandomFlip("horizontal_and_vertical") 
output = flip(image).numpy().astype("uint8")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 5))
ax1.imshow(image)
ax1.set_title("original")
ax1.set_axis_off()
# flip
ax2.imshow(output)
ax2.set_title("flip")
ax2.set_axis_off()
plt.show()