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

import tensorflow as tf

import keras_cv
from keras_cv.utils import assert_matplotlib_installed

try:
    import matplotlib.pyplot as plt
except:
    plt = None


def plot_gallery(
    images,
    value_range,
    rows=3,
    cols=3,
    scale=2,
    path=None,
    show=None,
    transparent=True,
    dpi=60,
    legend_handles=None,
):
    """gallery_show shows a gallery of images.

    Args:
        images: a Tensor or NumPy array containing images to show in the gallery.
        value_range: value range of the images.
        rows: number of rows in the gallery to show.
        cols: number of columns in the gallery to show.
        scale: how large to scale the images in the gallery
        path: (Optional) path to save the resulting gallery to.
        show: (Optional) whether or not to show the gallery of images.
        transparent: (Optional) whether or not to give the image a transparent
            background.  Defaults to True.
        dpi: (Optional) the dpi to pass to matplotlib.savefig().  Defaults to `60`.
        legend_handles: (Optional) matplotlib.patches List of legend handles.
            I.e. passing: `[patches.Patch(color='red', label='mylabel')]` will produce
            a legend with a single red patch and the label 'mylabel'.
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
        ds_iter = iter(images)
        total = 0
        images = []
        while total < rows * cols:
            inputs = next(ds_iter)
            if isinstance(inputs, dict):
                x = inputs["images"]
            elif isinstance(inputs, tuple):
                x = inputs[0]
            else:
                x = inputs
            total += x.shape[0]
            images.append(x)
            boxes.append(y)
        images = tf.concatenate(images, axis=0)

    fig = plt.figure(figsize=(cols * scale, rows * scale))
    fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    plt.axis("off")

    if legend_handles is not None:
        fig.legend(handles=legend_handles, loc="lower center")

    images = keras_cv.utils.transform_value_range(
        images, original_range=value_range, target_range=(0, 255)
    )
    if isinstance(images, tf.Tensor):
        images = images.numpy()

    images = images.astype(int)
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            plt.subplot(rows, cols, index + 1)
            plt.imshow(images[index].astype("uint8"))
            plt.axis("off")
            plt.margins(x=0, y=0)

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
