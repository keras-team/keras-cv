"""
Title: Plot an image gallery
Author: [lukewood](https://lukewood.xyz)
Date created: 2022/10/16
Last modified: 2022/10/16
Description: Visualize ground truth and predicted bounding boxes for a given
             dataset.
"""

"""
Plotting images from a TensorFlow dataset is easy with KerasCV. Behold:
"""

import tensorflow as tf
import tensorflow_datasets as tfds

import keras_cv

train_ds = tfds.load(
    "cats_vs_dogs",
    split="train",
    with_info=False,
    shuffle_files=True,
)


def unpackage_tfds_inputs(inputs):
    return inputs["image"]


train_ds = train_ds.map(unpackage_tfds_inputs)
train_ds = train_ds.apply(tf.data.experimental.dense_to_ragged_batch(16))

keras_cv.visualization.plot_image_gallery(
    next(iter(train_ds.take(1))),
    value_range=(0, 255),
    scale=3,
    rows=2,
    cols=2,
)
