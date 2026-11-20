from keras.utils import image_dataset_from_directory

ds = image_dataset_from_directory("/Dataset/cats_dogs_light/train",
                          validation_split=0.2, subset="training",
                          image_size=(256,256), interpolation="bilinear",
                          crop_to_aspect_ratio=True,
                          seed=42, shuffle=True, batch_size=32)

from keras.layers import CenterCrop, Rescaling
from keras import Sequential

height, width = 128, 128
centercrop = Sequential([
    CenterCrop(height, width),
    Rescaling(1.0/255)   
])
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(4,4))

for images, labels in ds.take(1):
    for i in range(2):
        ax[0][i].imshow(images[i])
        ax[0][i].set_title("original")
        ax[0][i].set_axis_off()
        # 呼叫 resize()
        ax[1][i].imshow(centercrop(images[i]))
        print(centercrop(images[i]))
        ax[1][i].set_title("centercroped")
        ax[1][i].set_axis_off()
plt.show()