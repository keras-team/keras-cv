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

import os
import re

import keras_cv  # noqa: E402

BUCKET = "keras-cv-kaggle"


def to_snake_case(name):
    name = re.sub(r"\W+", "", name)
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub("([a-z])([A-Z])", r"\1_\2", name).lower()
    return name


def convert_backbone_presets():
    # Save and upload Backbone presets

    backbone_models = [
        keras_cv.models.ResNetBackbone,
        keras_cv.models.ResNet18Backbone,
        keras_cv.models.ResNet34Backbone,
        keras_cv.models.ResNet50Backbone,
        keras_cv.models.ResNet101Backbone,
        keras_cv.models.ResNet152Backbone,
        keras_cv.models.ResNetV2Backbone,
        keras_cv.models.ResNet18V2Backbone,
        keras_cv.models.ResNet34V2Backbone,
        keras_cv.models.ResNet50V2Backbone,
        keras_cv.models.ResNet101V2Backbone,
        keras_cv.models.ResNet152V2Backbone,
        keras_cv.models.YOLOV8Backbone,
        keras_cv.models.MobileNetV3Backbone,
        keras_cv.models.MobileNetV3SmallBackbone,
        keras_cv.models.MobileNetV3LargeBackbone,
        keras_cv.models.EfficientNetV2Backbone,
        keras_cv.models.EfficientNetV2B0Backbone,
        keras_cv.models.EfficientNetV2B1Backbone,
        keras_cv.models.EfficientNetV2B2Backbone,
        keras_cv.models.EfficientNetV2B3Backbone,
        keras_cv.models.EfficientNetV2SBackbone,
        keras_cv.models.EfficientNetV2MBackbone,
        keras_cv.models.EfficientNetV2LBackbone,
        keras_cv.models.CSPDarkNetBackbone,
        keras_cv.models.DenseNetBackbone,
        keras_cv.src.models.EfficientNetV1Backbone,
        keras_cv.src.models.EfficientNetLiteBackbone,
        keras_cv.models.MiTBackbone,
        keras_cv.models.ViTDetBackbone,
        keras_cv.models.CenterPillarBackbone,
    ]
    for backbone_cls in backbone_models:
        for preset in backbone_cls.presets:
            backbone = backbone_cls.from_preset(
                preset, name=to_snake_case(backbone_cls.__name__)
            )
            save_weights = preset in backbone_cls.presets_with_weights
            save_to_preset(
                backbone,
                preset,
                save_weights=save_weights,
                config_filename="config.json",
            )
            # Delete first to clean up any exising version.
            os.system(f"gsutil rm -rf gs://{BUCKET}/{preset}")
            os.system(f"gsutil cp -r {preset} gs://{BUCKET}/{preset}")
            for root, _, files in os.walk(preset):
                for file in files:
                    path = os.path.join(BUCKET, root, file)
                    os.system(
                        f"gcloud storage objects update gs://{path} "
                        "--add-acl-grant=entity=AllUsers,role=READER"
                    )


def convert_task_presets():
    # Save and upload task presets

    task_models = [
        keras_cv.models.RetinaNet,
        keras_cv.models.YOLOV8Detector,
        keras_cv.models.ImageClassifier,
        keras_cv.models.DeepLabV3Plus,
        # keras_cv.models.SegFormer,
        keras_cv.models.SegmentAnythingModel,
    ]
    for task_cls in task_models:
        # Remove backbone-specific keys
        task_preset_keys = set(task_cls.presets) ^ set(
            task_cls.backbone_presets
        )
        for preset in task_preset_keys:
            save_weights = preset in task_cls.presets_with_weights
            kwargs = {"name": to_snake_case(task_cls.__name__)}
            if task_cls in [
                keras_cv.models.RetinaNet,
                keras_cv.models.YOLOV8Detector,
            ]:
                kwargs.update({"bounding_box_format": "xywh"})
                task = task_cls.from_preset(preset, **kwargs)
            else:
                task = task_cls.from_preset(preset, **kwargs)
            save_to_preset(
                task,
                preset,
                save_weights=save_weights,
                config_filename="config.json",
            )
            # Delete first to clean up any exising version.
            os.system(f"gsutil rm -rf gs://{BUCKET}/{preset}")
            os.system(f"gsutil cp -r {preset} gs://{BUCKET}/{preset}")
            for root, _, files in os.walk(preset):
                for file in files:
                    path = os.path.join(BUCKET, root, file)
                    os.system(
                        f"gcloud storage objects update gs://{path} "
                        "--add-acl-grant=entity=AllUsers,role=READER"
                    )


if __name__ == "__main__":
    from keras_cv.src.utils.preset_utils import save_to_preset  # noqa: E402

    convert_backbone_presets()
    convert_task_presets()
