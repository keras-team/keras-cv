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
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

from keras_cv import bounding_box


def unpackage_pascalvoc(bounding_box_format, img_size):
    """mapping function to create batched image and bbox coordinates"""

    resizing = keras.layers.Resizing(
        height=img_size[0], width=img_size[1], crop_to_aspect_ratio=False
    )

    def apply(inputs):
        inputs["image"] = resizing(inputs["image"])
        inputs["objects"]["bbox"] = bounding_box.convert_format(
            inputs["objects"]["bbox"],
            images=inputs["image"],
            source="rel_yxyx",
            target=bounding_box_format,
        )

        bounding_boxes = inputs["objects"]["bbox"]
        labels = tf.cast(inputs["objects"]["label"], tf.float32)
        labels = tf.expand_dims(labels, axis=-1)
        bounding_boxes = tf.concat([bounding_boxes, labels], axis=-1)
        return {"images": inputs["image"], "bounding_boxes": bounding_boxes}

    return apply


def load_pascal_voc(
    split, bounding_box_format, batch_size, shuffle=True, img_size=(512, 512)
):
    dataset, dataset_info = tfds.load(
        "voc/2007", split=split, shuffle_files=shuffle, with_info=True
    )
    dataset = dataset.map(
        unpackage_pascalvoc(bounding_box_format=bounding_box_format, img_size=img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size)
    )
    return dataset, dataset_info


def main():
    batch_size = 9
    dataset, ds_info = load_pascal_voc(
        split="train", bounding_box_format="rel_yxyx", batch_size=batch_size
    )

    for example in dataset.take(1):
        images, boxes = example["images"], example["bounding_boxes"]
        boxes = boxes.to_tensor(default_value=-1)
        print(boxes)
        color = tf.constant(((255.0, 0, 0),))
        plotted_images = tf.image.draw_bounding_boxes(
            images, boxes[..., :4], color, name=None
        )
        plt.figure(figsize=(10, 10))
        for i in range(batch_size):
            plt.subplot(batch_size // 3, batch_size // 3, i + 1)
            plt.imshow(plotted_images[i].numpy().astype("uint8"))
            plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
