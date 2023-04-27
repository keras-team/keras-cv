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
"""
Small utility script to count parameters in our preset checkpoints.

Usage:
python tools/count_preset_params.py
python tools/count_preset_params.py --model ResNetV2Backbone
python tools/count_preset_params.py --preset resnet50_v2_imagenet
"""

import inspect

from absl import app
from absl import flags
from keras.utils.layer_utils import count_params

import keras_cv

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "model", None, "The name of a model, e.g. ResNetV2Backbone."
)
flags.DEFINE_string(
    "preset", None, "The name of a preset, e.g. resnet50_v2_imagenet."
)


def main(_):
    for name, symbol in keras_cv.models.__dict__.items():
        if FLAGS.model and name != FLAGS.model:
            continue
        if not hasattr(symbol, "from_preset"):
            continue
        if not inspect.isclass(symbol):
            continue
        if not issubclass(
            symbol,
            (
                keras_cv.models.backbones.backbone.Backbone,
                keras_cv.models.task.Task,
            ),
        ):
            continue
        for preset in symbol.presets:
            if FLAGS.preset and preset != FLAGS.preset:
                continue

            # Avoid printing all backbone presets of each task.
            if issubclass(symbol, keras_cv.models.task.Task) and (
                preset
                in keras_cv.models.backbones.backbone_presets.backbone_presets
            ):
                continue

            if symbol in (
                keras_cv.models.RetinaNet,
                keras_cv.models.YOLOV8Detector,
            ):
                model = symbol.from_preset(preset, bounding_box_format="xywh")
            else:
                model = symbol.from_preset(preset)
            params = count_params(model.weights)
            print(f"{name} {preset} {params}")


if __name__ == "__main__":
    app.run(main)
