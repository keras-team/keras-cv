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
import keras_cv
import tensorflow as tf
from keras_cv import utils
from keras_cv.utils import assert_matplotlib_installed

try:
    import matplotlib.pyplot as plt
except:
    plt = None


def plot_image_gallery(
    images,
    value_range,
    rows=None,
    cols=None,
    batch_size=8,
    scale=2,
    path=None,
    show=None,
    transparent=True,
    dpi=60,
    legend_handles=None,
    image_key="image",
):
    """Displays a gallery of images.

    Usage:
    ```python
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
    ```

    ![example gallery](https://i.imgur.com/r0ndse0.png)

    Args:
        images: a Tensor or NumPy array containing images to show in the
            gallery.
        value_range: value range of the images. Common examples include
            `(0, 255)` and `(0, 1)`.
        scale: how large to scale the images in the gallery
        rows: (Optional) number of rows in the gallery to show.
        cols: (Optional) number of columns in the gallery to show.
        path: (Optional) path to save the resulting gallery to.
        show: (Optional) whether to show the gallery of images.
        transparent: (Optional) whether to give the image a transparent
            background, defaults to `True`.
        dpi: (Optional) the dpi to pass to matplotlib.savefig(), defaults to
            `60`.
        legend_handles: (Optional) matplotlib.patches List of legend handles.
            I.e. passing: `[patches.Patch(color='red', label='mylabel')]` will
            produce a legend with a single red patch and the label 'mylabel'.
        image_key: (Optional) Key of the argument holding the image. Only
            required when using a `tf.data.Dataset` instance. Defaults to
            "image".
    """
    assert_matplotlib_installed("plot_bounding_box_gallery")

    if path is None and show is None:
        # Default to showing the image
        show = True
    if path is not None and show:
        raise ValueError(
            "plot_gallery() expects either `path` to be set, or `show` "
            "to be true."
        )

    if isinstance(images, tf.data.Dataset):

        # Find final dataset batch size
        sample = next(iter(images.take(1)))
        if len(sample[image_key].shape) == 3:
            default_dataset_batch_size = 8
            images = images.ragged_batch(batch_size=default_dataset_batch_size)
        elif len(sample[image_key].shape) == 4:
            default_dataset_batch_size = sample[image_key].shape[0]
        else:
            raise ValueError(
                "plot_image_gallery() expects `tf.data.Dataset` to have TensorShape with length 3 or 4."
            )

        batches = default_dataset_batch_size

        def unpack_images(inputs):
            return inputs[image_key]

        images = images.map(unpack_images)
        images = images.take(batches)
        images = next(iter(images))

    # Calculate appropriate number of rows and columns
    if rows is None and cols is None:
        total_plots = batch_size
        cols = batch_size // 2

        rows = total_plots // cols

        if total_plots % cols != 0:
            rows += 1

    # Generate subplots
    fig, axes = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=(cols * scale, rows * scale),
        layout="tight",
        squeeze=True,
        sharex="row",
        sharey="col",
    )
    fig.subplots_adjust(wspace=0, hspace=0)

    if legend_handles is not None:
        fig.legend(handles=legend_handles, loc="lower center")

    # Perform image range transform
    images = keras_cv.utils.transform_value_range(
        images, original_range=value_range, target_range=(0, 255)
    )

    images = utils.to_numpy(images)
    images = images.astype(int)

    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            current_axis = axes[row, col]
            current_axis.imshow(images[index].astype("uint8"))
            current_axis.margins(x=0, y=0)
            current_axis.axis("off")

    if path is None and not show:
        return
    if path is not None:
        plt.savefig(
            fname=path,
            pad_inches=0,
            bbox_inches="tight",
            transparent=transparent,
            dpi=dpi,
        )
        plt.close()
    elif show:
        plt.show()
        plt.close()
