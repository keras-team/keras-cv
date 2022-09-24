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
import demo_utils
import tensorflow as tf

import keras_cv
from keras_cv import layers
from luketils import visualization

IMG_SIZE = (256, 256)
BATCH_SIZE = 9

class_ids = [
    "Aeroplane",
    "Bicycle",
    "Bird",
    "Boat",
    "Bottle",
    "Bus",
    "Car",
    "Cat",
    "Chair",
    "Cow",
    "Dining Table",
    "Dog",
    "Horse",
    "Motorbike",
    "Person",
    "Potted Plant",
    "Sheep",
    "Sofa",
    "Train",
    "Tvmonitor",
    "Total",
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))


def main():
    dataset, ds_info = keras_cv.datasets.pascal_voc.load(
        split="train", bounding_box_format="rel_xyxy", batch_size=1, shuffle=True
    )
    random_shear = layers.RandomCropAndResize(
        target_size=(512, 512),
        crop_area_factor=(1.0, 1.0),
        aspect_ratio_factor=(1.0, 1.0),
        bounding_box_format="rel_xyxy",
    )
    dataset = dataset.map(lambda x: random_shear(x, training=True), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda inputs: (inputs["images"], inputs["bounding_boxes"]))
    images, boxes = next(iter(dataset.take(1)))
    visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        y_true=boxes,
        bounding_box_format="rel_xyxy",
        class_mapping=class_mapping,
        rows=1,
        cols=1,
        show=True,
    )


if __name__ == "__main__":
    main()
