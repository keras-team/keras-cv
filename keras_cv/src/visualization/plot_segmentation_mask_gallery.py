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

from keras_cv.src import utils
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.utils import assert_matplotlib_installed
from keras_cv.src.visualization.plot_image_gallery import plot_image_gallery


def reshape_masks(segmentation_masks):
    rank = len(segmentation_masks.shape)
    if rank == 3:
        # (B, H, W)
        return segmentation_masks[..., np.newaxis]
    elif rank == 4:
        # (B, H, W, num_channels) OR (B, H, W, 1)
        if segmentation_masks.shape[-1] == 1:
            # Repeat the masks 3 times in order to build 3 channel
            # segmentation masks.
            return segmentation_masks.repeat(repeats=3, axis=-1)
        else:
            return np.argmax(segmentation_masks, axis=-1).repeat(
                repeats=3, axis=-1
            )


def transform_segmentation_masks(segmentation_masks, num_classes, value_range):
    segmentation_masks = utils.to_numpy(segmentation_masks)
    segmentation_masks = reshape_masks(segmentation_masks=segmentation_masks)

    # Interpolate the segmentation masks from the range of (0, num_classes)
    # to the value range provided.
    segmentation_masks = utils.transform_value_range(
        segmentation_masks,
        original_range=(0, num_classes),
        target_range=value_range,
    )
    return segmentation_masks


@keras_cv_export("keras_cv.visualization.plot_segmentation_mask_gallery")
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

    Args:
        images: a Tensor or NumPy array containing images to show in the
            gallery. The images should be batched and of shape (B, H, W, C).
        value_range: value range of the images. Common examples include
            `(0, 255)` and `(0, 1)`.
        num_classes: number of segmentation classes.
        y_true: (Optional) a Tensor or NumPy array representing the ground truth
            segmentation masks. The ground truth segmentation maps should be
            batched.
        y_pred: (Optional)  a Tensor or NumPy array representing the predicted
            segmentation masks. The predicted segmentation masks should be
            batched.
        kwargs: keyword arguments to propagate to
            `keras_cv.visualization.plot_image_gallery()`.

    Example:
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
        y_pred=None,
        scale=3,
        rows=2,
        cols=2,
    )
    ```

    ![Example segmentation mask gallery](https://i.imgur.com/aRkmJ1Q.png)
    """
    assert_matplotlib_installed("plot_segmentation_mask_gallery")

    plotted_images = utils.to_numpy(images)

    # Initialize a list to collect the segmentation masks that will be
    # concatenated to the images for visualization.
    masks_to_contatenate = [plotted_images]

    if y_true is not None:
        plotted_y_true = transform_segmentation_masks(
            segmentation_masks=y_true,
            num_classes=num_classes,
            value_range=value_range,
        )
        masks_to_contatenate.append(plotted_y_true)
    if y_pred is not None:
        plotted_y_pred = transform_segmentation_masks(
            segmentation_masks=y_pred,
            num_classes=num_classes,
            value_range=value_range,
        )
        masks_to_contatenate.append(plotted_y_pred)

    # Concatenate the images and the masks together.
    plotted_images = np.concatenate(masks_to_contatenate, axis=2)

    plot_image_gallery(
        plotted_images, value_range, rows=rows, cols=cols, **kwargs
    )
