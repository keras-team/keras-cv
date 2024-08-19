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

import math

import numpy as np
import tensorflow as tf

from keras_cv.src import utils
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import ops
from keras_cv.src.utils import assert_matplotlib_installed

try:
    import matplotlib.pyplot as plt
except:
    plt = None


def _extract_image_batch(images, num_images, batch_size):
    def unpack_images(inputs):
        return inputs["image"]

    num_batches_required = math.ceil(num_images / batch_size)

    if isinstance(images, tf.data.Dataset):
        images = images.map(unpack_images)

        if batch_size == 1:
            images = images.ragged_batch(num_batches_required)
            sample = next(iter(images.take(1)))
        else:
            sample = next(iter(images.take(num_batches_required)))

        return sample

    else:
        if len(ops.shape(images)) != 4:
            raise ValueError(
                "`plot_images_gallery()` requires you to "
                "batch your `np.array` samples together."
            )
        else:
            num_samples = (
                num_images if num_images <= batch_size else num_batches_required
            )
            sample = images[:num_samples, ...]
    return sample


@keras_cv_export("keras_cv.visualization.plot_image_gallery")
def plot_image_gallery(
    images,
    value_range,
    scale=2,
    rows=None,
    cols=None,
    path=None,
    show=None,
    transparent=True,
    dpi=60,
    legend_handles=None,
):
    """Displays a gallery of images.

    Example:
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
        images: a Tensor, `tf.data.Dataset` or NumPy array containing images
            to show in the gallery. Note: If using a `tf.data.Dataset`,
            images should be present in the `FeaturesDict` under
            the key `image`.
        value_range: value range of the images. Common examples include
            `(0, 255)` and `(0, 1)`.
        scale: how large to scale the images in the gallery
        rows: (Optional) number of rows in the gallery to show.
            Required if inputs are unbatched.
        cols: (Optional) number of columns in the gallery to show.
            Required if inputs are unbatched.
        path: (Optional) path to save the resulting gallery to.
        show: (Optional) whether to show the gallery of images.
        transparent: (Optional) whether to give the image a transparent
            background, defaults to `True`.
        dpi: (Optional) the dpi to pass to matplotlib.savefig(), defaults to
            `60`.
        legend_handles: (Optional) matplotlib.patches List of legend handles.
            I.e. passing: `[patches.Patch(color='red', label='mylabel')]` will
            produce a legend with a single red patch and the label 'mylabel'.

    """
    assert_matplotlib_installed("plot_bounding_box_gallery")

    if path is not None and show:
        raise ValueError(
            "plot_gallery() expects either `path` to be set, or `show` "
            "to be true."
        )

    if isinstance(images, tf.data.Dataset):
        sample = next(iter(images.take(1)))
        batch_size = (
            sample["image"].shape[0] if len(sample["image"].shape) == 4 else 1
        )  # batch_size from within passed `tf.data.Dataset`
    else:
        batch_size = (
            ops.shape(images)[0] if len(ops.shape(images)) == 4 else 1
        )  # batch_size from np.array or single image

    rows = rows or int(math.ceil(math.sqrt(batch_size)))
    cols = cols or int(math.ceil(batch_size // rows))

    num_images = rows * cols
    images = _extract_image_batch(images, num_images, batch_size)

    # Generate subplots
    fig, axes = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=(cols * scale, rows * scale),
        frameon=False,
        layout="tight",
        squeeze=True,
        sharex="row",
        sharey="col",
    )
    fig.subplots_adjust(wspace=0, hspace=0)

    if isinstance(axes, np.ndarray) and len(axes.shape) == 1:
        expand_axis = 0 if rows == 1 else -1
        axes = np.expand_dims(axes, expand_axis)

    if legend_handles is not None:
        fig.legend(handles=legend_handles, loc="lower center")

    # Perform image range transform
    images = utils.transform_value_range(
        images, original_range=value_range, target_range=(0, 255)
    )
    images = utils.to_numpy(images)

    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            current_axis = (
                axes[row, col] if isinstance(axes, np.ndarray) else axes
            )
            current_axis.imshow(images[index].astype("uint8"))
            current_axis.margins(x=0, y=0)
            current_axis.axis("off")

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
    else:
        return fig
