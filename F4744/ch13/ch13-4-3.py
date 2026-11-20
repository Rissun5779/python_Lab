from keras.utils import image_dataset_from_directory

ds = image_dataset_from_directory("/Dataset/cats_dogs_light/train",
                          validation_split=0.2, subset="training",
                          image_size=(256,256), interpolation="bilinear",
                          crop_to_aspect_ratio=True,
                          seed=42, shuffle=True, batch_size=32)

from keras.layers import Resizing

height, width = 128, 256
resize = Resizing(height, width)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(4,4))

for images, labels in ds.take(1):
    for i in range(2):
        ax[0][i].imshow(images[i])
        ax[0][i].set_title("original")
        ax[0][i].set_axis_off()
        # 呼叫 resize()
        ax[1][i].imshow(resize(images[i]))
        ax[1][i].set_title("resized")
        ax[1][i].set_axis_off()
plt.show()