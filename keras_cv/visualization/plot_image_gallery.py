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
        images: a Tensor, `tf.data.Dataset` or NumPy array containing images to show in the
            gallery.
        value_range: value range of the images. Common examples include
            `(0, 255)` and `(0, 1)`.
        scale: how large to scale the images in the gallery
        rows: (Optional) number of rows in the gallery to show. Required if inputs are unbatched.
        cols: (Optional) number of columns in the gallery to show. Required if inputs are unbatched.
        path: (Optional) path to save the resulting gallery to.
        show: (Optional) whether to show the gallery of images.
        transparent: (Optional) whether to give the image a transparent
            background, defaults to `True`.
        dpi: (Optional) the dpi to pass to matplotlib.savefig(), defaults to
            `60`.
        legend_handles: (Optional) matplotlib.patches List of legend handles.
            I.e. passing: `[patches.Patch(color='red', label='mylabel')]` will
            produce a legend with a single red patch and the label 'mylabel'.

    Note:
        If using a `tf.data.Dataset`, it is important that the images present in
        the `FeaturesDict` should have the key `image`.
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

    def unpack_images(inputs):
        return inputs["image"]

    # Calculate appropriate number of rows and columns
    if rows is None or cols is None:
        if isinstance(images, tf.data.Dataset):
            sample = next(iter(images.take(1)))
            sample_shape = sample["image"].shape
            if len(sample_shape) == 4:
                batch_size = sample_shape[0]
            else:
                raise ValueError(
                    "Passed `tf.data.Dataset` does not appear to be batched. Please batch using the `.batch().`"
                )

            images = images.map(unpack_images)
            images = images.take(batch_size)
            images = next(iter(images))
        else:
            sample_shape = images.shape
            if len(sample_shape) == 4:
                batch_size = sample_shape[0]
            else:
                raise ValueError(
                    f"`plot_image_gallery` received unbatched images and `cols` and `rows` "
                    "were both `None`. Either images should be batched, or `cols` and `rows` should be specified."
                )

    elif rows is not None and cols is not None:
        if isinstance(images, tf.data.Dataset):
            batch_size = rows * cols

            sample = next(iter(images.take(1)))
            sample_shape = sample["image"].shape

            if len(sample_shape) == 4:
                images = images.unbatch()

            images = images.ragged_batch(batch_size=batch_size)

            images = images.map(unpack_images)
            images = images.take(batch_size)
            images = next(iter(images))
        else:
            batch_size = rows * cols
            images = images[:batch_size, ...]
    else:
        raise ValueError(
            "plot_image_gallery() expects `tf.data.Dataset` to be batched if rows or cols are not specified."
        )

    rows = int(math.ceil(batch_size**0.5))
    cols = int(math.ceil(batch_size // rows))

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
