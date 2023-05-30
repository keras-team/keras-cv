"""
Title: Plot an image gallery
Author: [lukewood](https://lukewood.xyz)
Date created: 2022/10/16
Last modified: 2022/05/31
Description: Visualize ground truth and predicted bounding boxes for a given
             dataset.
"""

"""
Plotting images from a TensorFlow dataset is easy with KerasCV. Behold:
"""

import tensorflow_datasets as tfds

import keras_cv

train_ds = tfds.load(
    "cats_vs_dogs",
    split="train",
    with_info=False,
    shuffle_files=True,
)

train_ds = train_ds.ragged_batch(16)

keras_cv.visualization.plot_image_gallery(
    train_ds,
    value_range=(0, 255),
    scale=3,
)
