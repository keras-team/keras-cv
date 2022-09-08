# Copyright 2022 The KerasCV Authors
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


import keras_cv
import tensorflow as tf


def plot_gallery(images, value_range, rows=3, columns=3, scale=2, path=None):
    """gallery_show shows a gallery of images.

    Args:
        images: a Tensor or NumPy array containing images to show in the gallery.
        value_range: value range of the images.
        rows: number of rows in the gallery to show.
        columns: number of columns in the gallery to show.
        scale: how large to scale the images in the gallery
        path: (Optional) path to save the resulting gallery to.
    """
    plt = keras_cv.visualization.get_plt()
    fig = plt.figure(figsize=(columns * scale, rows * scale))
    fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    plt.axis("off")

    images = keras_cv.utils.transform_value_range(images, original_range=value_range, target_range=(0, 255))
    if isinstance(images, tf.Tensor):
        images = images.numpy()

    images = images.astype(int)
    for row in range(rows):
        for col in range(columns):
            index = row * columns + col
            plt.subplot(rows, columns, index + 1)
            plt.imshow(images[index].astype("uint8"))
            plt.axis("off")
            plt.margins(x=0, y=0)

    if path is not None:
        plt.savefig(fname=path, pad_inches=0, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
