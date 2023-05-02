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
"""Temporary file for copying weights to the new CSPDarkNetBackbone."""

import tensorflow as tf

from keras_cv.models import YOLOV8Backbone
from keras_cv.models.backbones.csp_darknet import csp_darknet_backbone
from keras_cv.models.backbones.csp_darknet import legacy
from keras_cv.models.object_detection.yolo_v8.yolo_v8_backbone_presets import (
    copy_weights,
)

yolo_presets = [
    ("xs", "yolov8_xs_backbone_coco", "yolov8_xs_backbone"),
    ("s", "yolov8_s_backbone_coco", "yolov8_s_backbone"),
    ("m", "yolov8_m_backbone_coco", "yolov8_m_backbone"),
    ("l", "yolov8_l_backbone_coco", "yolov8_l_backbone"),
    ("xl", "yolov8_xl_backbone_coco", "yolov8_xl_backbone"),
]
for name, yolo_preset, csp_preset in yolo_presets:
    yolo_model = YOLOV8Backbone.from_preset(yolo_preset)
    csp_model = csp_darknet_backbone.CSPDarkNetBackbone.from_preset(csp_preset)
    copy_weights(yolo_model, csp_model)
    outputs = csp_model(tf.ones(shape=(1, 512, 512, 3)))
    expected = yolo_model(tf.ones(shape=(1, 512, 512, 3)))
    assert tf.math.reduce_all(tf.math.equal(outputs, expected))
    csp_model.save_weights(f"new_csp_weights/{yolo_preset}.h5")

csp_presets = [
    ("tiny", "csp_darknet_tiny_imagenet", "csp_darknet_tiny"),
    ("l", "csp_darknet_l_imagenet", "csp_darknet_l"),
]
for name, old_csp_preset, csp_preset in csp_presets:
    old_csp_model = legacy.csp_darknet_backbone.CSPDarkNetBackbone.from_preset(
        old_csp_preset
    )
    csp_model = csp_darknet_backbone.CSPDarkNetBackbone.from_preset(csp_preset)
    copy_weights(old_csp_model, csp_model)
    outputs = csp_model(tf.ones(shape=(1, 512, 512, 3)))
    expected = old_csp_model(tf.ones(shape=(1, 512, 512, 3)))
    assert tf.math.reduce_all(tf.math.equal(outputs, expected))
    csp_model.save_weights(f"new_csp_weights/{old_csp_preset}.h5")
