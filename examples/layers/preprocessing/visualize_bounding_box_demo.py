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
"""
Visualize bounding box demo illustrates how to use
visualize_bounding_boxes_on_image
"""
import numpy as np
import tensorflow as tf

from keras_cv import bounding_box

img = tf.zeros([20, 20, 3])
bboxes = np.array([[0, 0, 10, 10], [4, 4, 12, 12]])
bboxes = tf.convert_to_tensor(bboxes, dtype=tf.int32)
bounding_box.visualize_bounding_boxes_on_image(img, bboxes, bounding_box_format="xyxy")
