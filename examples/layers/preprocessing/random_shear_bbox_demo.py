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
"""random_shear_bboxdemo.py shows how to use the RandomShear preprocessing layer
   for object detection.
Operates on the voc dataset.  In this script the images and bboxes
are loaded, then are passed through the preprocessing layers.
Finally, they are shown using matplotlib.
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from keras_cv.bounding_box import convert_format
from keras_cv.layers import preprocessing

IMG_SIZE = (256, 256)
BATCH_SIZE = 9


def resize(inputs):
    """mapping function to create batched image and bbox coordinates"""
    inputs["image"] = tf.image.resize(inputs["image"], IMG_SIZE)[0]
    inputs["objects"]["bbox"] = tf.squeeze(
        convert_format(
            inputs["objects"]["bbox"],
            images=inputs["image"],
            source="rel_yxyx",
            target="rel_xyxy",
        ),
        axis=0,
    )
    return inputs


def main():
    dataset = tfds.load(
        "voc/2007", split=tfds.Split.TRAIN, batch_size=1, shuffle_files=True
    )
    dataset = dataset.map(lambda x: resize(x))
    dataset = dataset.padded_batch(BATCH_SIZE)

    randomshear = preprocessing.RandomShear(
        x_factor=(0.1, 0.3),
        y_factor=(0.1, 0.3),
        fill_mode="constant",
        bounding_box_format="rel_xyxy",
    )
    colors = np.array([[0.0, 255.0, 0.0]])
    for example in dataset.take(4):
        result = randomshear(
            {"images": example["image"], "bounding_boxes": example["objects"]["bbox"]}
        )
        images, bboxes = result["images"], result["bounding_boxes"]
        rel_yxyx_bboxes = convert_format(
            bboxes,
            source="rel_xyxy",
            target="rel_yxyx",
            images=images,
            dtype=bboxes.dtype,
        )
        plotted_images = tf.image.draw_bounding_boxes(
            images, rel_yxyx_bboxes, colors, name=None
        )
        plt.figure(figsize=(20, 20))
        for i in range(BATCH_SIZE):
            plt.subplot(BATCH_SIZE // 3, BATCH_SIZE // 3, i + 1)
            plt.imshow(plotted_images[i].numpy().astype("uint8"))
            plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
