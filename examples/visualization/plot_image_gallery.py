"""
Title: Plot an image gallery
Author: [lukewood](https://lukewood.xyz), updated by
[Suvaditya Mukherjee](https://twitter.com/halcyonrayes)
Date created: 2022/10/16
Last modified: 2022/06/24
Description: Visualize ground truth and predicted bounding boxes for a given
            dataset.
"""

"""
Plotting images from a TensorFlow dataset is easy with KerasCV. Behold:
"""

import numpy as np
import tensorflow_datasets as tfds

import keras_cv

train_ds = tfds.load(
    "cats_vs_dogs",
    split="train",
    with_info=False,
    shuffle_files=True,
)

keras_cv.visualization.plot_image_gallery(
    train_ds,
    value_range=(0, 255),
    scale=3,
)

"""
If you want to use plain NumPy arrays, you can do that too:
"""

# Prepare some sample NumPy arrays from random noise

samples = np.random.randint(0, 255, (20, 224, 224, 3))

keras_cv.visualization.plot_image_gallery(
    samples, value_range=(0, 255), scale=3, rows=4, cols=5
)
