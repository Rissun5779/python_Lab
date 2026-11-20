from keras.utils import image_dataset_from_directory
from keras.layers import RandomBrightness, RandomContrast
import matplotlib.pyplot as plt

ds = image_dataset_from_directory("/Dataset/cats_dogs_light/train",
                          validation_split=0.2, subset="training",
                          image_size=(256,256), interpolation="bilinear",
                          crop_to_aspect_ratio=True,
                          seed=42, shuffle=True, batch_size=32)

brightness = RandomBrightness((-0.8,0.8))
contrast = RandomContrast(0.2)

fig, ax = plt.subplots(3, 2, figsize=(6,7))

for images, labels in ds.take(1):
    for i in range(2):
        ax[0][i].imshow(images[i].numpy().astype("uint8"))
        ax[0][i].set_title("original")
        ax[0][i].set_axis_off()
        # brightness
        ax[1][i].imshow(brightness(images[i]).numpy().astype("uint8"))
        ax[1][i].set_title("brightness")
        ax[1][i].set_axis_off()
        # contrast
        ax[2][i].imshow(contrast(images[i]).numpy().astype("uint8"))
        ax[2][i].set_title("contrast")
        ax[2][i].set_axis_off()

plt.show()