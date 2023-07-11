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
import numpy as np
import tensorflow as tf

import keras_cv
from keras_cv.backend import ops


def _create_bounding_box_dataset(
    bounding_box_format, use_dictionary_box_format=False
):
    # Just about the easiest dataset you can have, all classes are 0, all boxes
    # are exactly the same. [1, 1, 2, 2] are the coordinates in xyxy.
    xs = np.random.normal(size=(1, 512, 512, 3))
    xs = ops.tile(xs, [5, 1, 1, 1])

    y_classes = ops.zeros((5, 3), "float32")

    ys = ops.array(
        [
            [0.1, 0.1, 0.23, 0.23],
            [0.67, 0.75, 0.23, 0.23],
            [0.25, 0.25, 0.23, 0.23],
        ],
        "float32",
    )
    ys = ops.expand_dims(ys, axis=0)
    ys = ops.tile(ys, [5, 1, 1])
    ys = keras_cv.bounding_box.convert_format(
        ys,
        source="rel_xywh",
        target=bounding_box_format,
        images=xs,
        dtype="float32",
    )
    num_dets = ops.ones([5])

    if use_dictionary_box_format:
        return tf.data.Dataset.from_tensor_slices(
            {
                "images": xs,
                "bounding_boxes": {
                    "boxes": ys,
                    "classes": y_classes,
                    "num_dets": num_dets,
                },
            }
        ).batch(5, drop_remainder=True)
    else:
        return xs, {"boxes": ys, "classes": y_classes}
