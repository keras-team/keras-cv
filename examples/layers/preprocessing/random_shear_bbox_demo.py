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
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from keras_cv.layers import preprocessing

IMG_SIZE = (500, 500)
BATCH_SIZE = 9


def resize(inputs):
    """mapping function to create batched image and bbox coordinates"""
    inputs["image"] = tf.image.resize(inputs["image"], IMG_SIZE)[0]
    height, width, _ = inputs["image"].shape
    y1, x1, y2, x2 = tf.split(inputs["objects"]["bbox"][0], 4, axis=1)
    inputs["objects"]["bbox"] = tf.squeeze(
        tf.stack(
            [
                y1 * (IMG_SIZE[0] / height),
                x1 * (IMG_SIZE[1] / width),
                y2 * (IMG_SIZE[0] / height),
                x2 * (IMG_SIZE[1] / width),
            ],
            axis=1,
        ),
        axis=-1,
    )
    return inputs


def plot_bbox(image, bbox):
    """plots bbox over image"""
    image = image.astype(np.uint8)
    h, w, _ = image.shape
    bbox[..., [0, 2]] = bbox[..., [0, 2]] * h
    bbox[..., [1, 3]] = bbox[..., [1, 3]] * w
    bbox = bbox.astype(int)
    for coor in bbox:
        y1, x1, y2, x2 = coor
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image


def main():
    dataset = tfds.load("voc/2007", split=tfds.Split.TRAIN, batch_size=1)
    dataset = dataset.map(lambda x: resize(x))
    dataset = dataset.padded_batch(BATCH_SIZE)

    randomshear = preprocessing.RandomShear(
        x_factor=(0, 0.5), y_factor=0.2, fill_mode="constant"
    )

    for example in iter(dataset):
        result = randomshear(
            {"images": example["image"], "bounding_boxes": example["objects"]["bbox"]}
        )
        images, bboxes = result["images"], result["bounding_boxes"]
        plt.figure(figsize=(20, 20))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            image = plot_bbox(images[i].numpy(), bboxes[i].numpy())
            plt.imshow(image.astype("uint8"))
            plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
