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
"""resize_demo.py shows how to use the Resizing preprocessing layer.

Uses the oxford iiit pet_dataset.  In this script the pets
are loaded, then are passed through the preprocessing layers.
Finally, they are shown using matplotlib.
"""
import tensorflow as tf
import tensorflow_datasets as tfds

from keras_cv.layers import preprocessing
from keras_cv.visualization import plot_image_gallery


def load_data():
    ds = tfds.load(
        name="oxford_iiit_pet",
        split="train",
    )
    return ds.map(
        lambda inputs: {
            "images": tf.cast(inputs["image"], dtype=tf.float32),
            "segmentation_masks": inputs["segmentation_mask"] - 1,
        }
    )


def map_fn_for_visualization(inputs):
    masks = tf.cast(inputs["segmentation_masks"], dtype=tf.float32) / 2.0

    images = tf.expand_dims(inputs["images"], axis=0)
    masks = tf.expand_dims(masks, axis=0)

    masks = tf.repeat(masks, repeats=3, axis=-1)
    image_masks = tf.concat([images, masks], axis=2)
    return image_masks[0]


def main():
    ds = load_data()
    resize = preprocessing.Resizing(
        256,
        256,
        interpolation="bilinear",
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=False,
        bounding_box_format=None,
    )
    resize_crop = preprocessing.Resizing(
        256,
        256,
        interpolation="bilinear",
        crop_to_aspect_ratio=True,
        pad_to_aspect_ratio=False,
        bounding_box_format=None,
    )
    resize_pad = preprocessing.Resizing(
        256,
        256,
        interpolation="bilinear",
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=True,
        bounding_box_format=None,
    )

    ds_resize = ds.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
    ds_crop = ds.map(resize_crop, num_parallel_calls=tf.data.AUTOTUNE)
    ds_pad = ds.map(resize_pad, num_parallel_calls=tf.data.AUTOTUNE)

    ds_resize = ds_resize.map(map_fn_for_visualization).batch(8)
    ds_crop = ds_crop.map(map_fn_for_visualization).batch(8)
    ds_pad = ds_pad.map(map_fn_for_visualization).batch(8)

    plot_image_gallery(
        next(iter(ds_resize.take(1))),
        value_range=(0, 1),
        scale=3,
        rows=2,
        cols=4,
        path="resize.png",
    )
    plot_image_gallery(
        next(iter(ds_crop.take(1))),
        value_range=(0, 1),
        scale=3,
        rows=2,
        cols=4,
        path="resize_crop.png",
    )
    plot_image_gallery(
        next(iter(ds_pad.take(1))),
        value_range=(0, 1),
        scale=3,
        rows=2,
        cols=4,
        path="resize_pad.png",
    )


if __name__ == "__main__":
    main()
