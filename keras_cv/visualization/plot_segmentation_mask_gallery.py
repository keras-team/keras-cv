# Copyright 2023 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from keras_cv import utils
from keras_cv.utils import assert_matplotlib_installed
from keras_cv.visualization.plot_image_gallery import plot_image_gallery


def plot_segmentation_mask_gallery(
    images,
    value_range,
    num_classes,
    y_true=None,
    y_pred=None,
    rows=3,
    cols=3,
    **kwargs
):
    """Plots a gallery of images with corresponding segmentation masks.

    Usage:
    ```python
    train_ds = tfds.load(
        "oxford_iiit_pet", split="train", with_info=False, shuffle_files=True
    )

    def unpackage_tfds_inputs(inputs):
        image = inputs["image"]
        segmentation_mask = inputs["segmentation_mask"]
        return image, segmentation_mask

    train_ds = train_ds.map(unpackage_tfds_inputs).ragged_batch(16)
    images, segmentation_masks = next(iter(train_ds.take(1)))

    keras_cv.visualization.plot_segmentation_mask_gallery(
        images,
        value_range=(0, 255),
        num_classes=3, # The number of classes for the oxford iiit pet dataset
        y_true=segmentation_masks,
        y_true=segmentation_masks,
        scale=3,
        rows=2,
        cols=2,
    )
    ```

    ![Example bounding box gallery](https://i.imgur.com/tJpb8hZ.png)

    Args:
        images: a Tensor or NumPy array containing images to show in the
            gallery.
        value_range: value range of the images. Common examples include
            `(0, 255)` and `(0, 1)`.
        num_classes: number of segmentation classes.
        y_true: (Optional) a Tensor or NumPy array representing the ground truth
            segmentation masks.
        y_pred: (Optional)  a Tensor or NumPy array representing the predicted
            segmentation masks.
        kwargs: keyword arguments to propagate to
            `keras_cv.visualization.plot_image_gallery()`.
    """
    assert_matplotlib_installed("plot_segmentation_mask_gallery")

    plotted_images = utils.to_numpy(images)

    # Segmentation maps are of 1 channel, here we repeat the channel 3 times
    # to mimic a 3 channel image.
    plotted_y_true = utils.to_numpy(y_true).repeat(repeats=3, axis=-1)

    # Interpolate the segmentation maps from the range of (0, num_classes)
    # to the value range provided.
    plotted_y_true = np.interp(plotted_y_true, (0, num_classes), value_range)

    if y_pred is not None:
        plotted_y_pred = utils.to_numpy(y_pred).repeat(repeats=3, axis=-1)
        plotted_y_pred = np.interp(
            plotted_y_pred, (0, num_classes), value_range
        )

        # Concatenate the image and the segmentation maps into a single image.
        plotted_images = np.concatenate(
            [plotted_images, plotted_y_true, plotted_y_pred], axis=2
        )
    else:
        plotted_images = np.concatenate(
            [plotted_images, plotted_y_true], axis=2
        )

    plot_image_gallery(
        plotted_images, value_range, rows=rows, cols=cols, **kwargs
    )
