"""mix_up_example.py shows how to use the RandomMixUp preprocessing layer to 
preprocess the oxford_flowers102 dataset.  In this script the flowers 
are loaded, then are passed through the preprocessing layers.  
Finally, they are shown using matplotlib.
"""
import tensorflow as tf
from keras_cv.layers.preprocessing.random_mix_up import RandomMixUp
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


IMG_SIZE = 224
BATCH = 16

AUTOTUNE = tf.data.AUTOTUNE


def resize(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


def main():
    data, ds_info = tfds.load("oxford_flowers102", with_info=True, as_supervised=True)
    train_ds = data["train"]

    num_classes = ds_info.features["label"].num_classes

    train_ds = (
        train_ds.map(resize).shuffle(10 * BATCH).batch(BATCH, drop_remainder=True)
    )
    mixup = RandomMixUp(num_classes=num_classes)
    train_ds = train_ds.map(lambda x, y: mixup((x, y)), num_parallel_calls=AUTOTUNE)

    for images, labels in train_ds.take(1):
        plt.figure(figsize=(8, 8))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
