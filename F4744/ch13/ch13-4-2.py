from keras.utils import image_dataset_from_directory

ds = image_dataset_from_directory("/Dataset/cats_dogs_light/train",
                          validation_split=0.2, subset="training",
                          image_size=(256,256), interpolation="bilinear",
                          crop_to_aspect_ratio=True,
                          seed=42, shuffle=True, batch_size=32)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(5,5))

for images, labels in ds.take(1):
    for i in range(3):
        for j in range(3):
            ax[i][j].imshow(images[i*3+j])
            ax[i][j].set_title(ds.class_names[labels[i*3+j]])
            ax[i][j].set_axis_off()
plt.show()