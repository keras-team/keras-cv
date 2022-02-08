"""random_cutout_demo.py shows how to use the RandomCutout preprocessing layer.

Operates on the oxford_flowers102 dataset.  In this script the flowers
are loaded, then are passed through the preprocessing layers.
Finally, they are shown using matplotlib.
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from keras_cv.layers import preprocessing

IMG_SIZE = (224, 224)
BATCH_SIZE = 64


def resize(image, label, num_classes=10):
    image = tf.image.resize(image, IMG_SIZE)
    label = tf.one_hot(label, num_classes)
    return image, label


def main():
    data, ds_info = tfds.load("oxford_flowers102", with_info=True, as_supervised=True)
    train_ds = data["train"]

    num_classes = ds_info.features["label"].num_classes

    train_ds = (
        train_ds.map(lambda x, y: resize(x, y, num_classes=num_classes))
        .shuffle(10 * BATCH_SIZE)
        .batch(BATCH_SIZE)
    )
    random_cutout = preprocessing.RandomCutout(
        height_factor=(0.3, 0.9),
        width_factor=64,
        fill_mode="gaussian_noise",
    )
    train_ds = train_ds.map(
        lambda x, y: (random_cutout(x), y), num_parallel_calls=tf.data.AUTOTUNE
    )

    for images, labels in train_ds.take(1):
        plt.figure(figsize=(8, 8))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
