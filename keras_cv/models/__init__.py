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

from keras_cv.models import legacy
from keras_cv.models.backbones.csp_darknet.csp_darknet_aliases import (
    CSPDarkNetLBackbone,
)
from keras_cv.models.backbones.csp_darknet.csp_darknet_aliases import (
    CSPDarkNetMBackbone,
)
from keras_cv.models.backbones.csp_darknet.csp_darknet_aliases import (
    CSPDarkNetSBackbone,
)
from keras_cv.models.backbones.csp_darknet.csp_darknet_aliases import (
    CSPDarkNetTinyBackbone,
)
from keras_cv.models.backbones.csp_darknet.csp_darknet_aliases import (
    CSPDarkNetXLBackbone,
)
from keras_cv.models.backbones.csp_darknet.csp_darknet_backbone import (
    CSPDarkNetBackbone,
)
from keras_cv.models.backbones.densenet.densenet_aliases import (
    DenseNet121Backbone,
)
from keras_cv.models.backbones.densenet.densenet_aliases import (
    DenseNet169Backbone,
)
from keras_cv.models.backbones.densenet.densenet_aliases import (
    DenseNet201Backbone,
)
from keras_cv.models.backbones.densenet.densenet_backbone import (
    DenseNetBackbone,
)
from keras_cv.models.backbones.efficientnet_lite.efficientnet_lite_aliases import (  # noqa: E501
    EfficientNetLiteB0Backbone,
)
from keras_cv.models.backbones.efficientnet_lite.efficientnet_lite_aliases import (  # noqa: E501
    EfficientNetLiteB1Backbone,
)
from keras_cv.models.backbones.efficientnet_lite.efficientnet_lite_aliases import (  # noqa: E501
    EfficientNetLiteB2Backbone,
)
from keras_cv.models.backbones.efficientnet_lite.efficientnet_lite_aliases import (  # noqa: E501
    EfficientNetLiteB3Backbone,
)
from keras_cv.models.backbones.efficientnet_lite.efficientnet_lite_aliases import (  # noqa: E501
    EfficientNetLiteB4Backbone,
)
from keras_cv.models.backbones.efficientnet_lite.efficientnet_lite_backbone import (  # noqa: E501
    EfficientNetLiteBackbone,
)
from keras_cv.models.backbones.efficientnet_v1.efficientnet_v1_aliases import (
    EfficientNetV1B0Backbone,
)
from keras_cv.models.backbones.efficientnet_v1.efficientnet_v1_aliases import (
    EfficientNetV1B1Backbone,
)
from keras_cv.models.backbones.efficientnet_v1.efficientnet_v1_aliases import (
    EfficientNetV1B2Backbone,
)
from keras_cv.models.backbones.efficientnet_v1.efficientnet_v1_aliases import (
    EfficientNetV1B3Backbone,
)
from keras_cv.models.backbones.efficientnet_v1.efficientnet_v1_aliases import (
    EfficientNetV1B4Backbone,
)
from keras_cv.models.backbones.efficientnet_v1.efficientnet_v1_aliases import (
    EfficientNetV1B5Backbone,
)
from keras_cv.models.backbones.efficientnet_v1.efficientnet_v1_aliases import (
    EfficientNetV1B6Backbone,
)
from keras_cv.models.backbones.efficientnet_v1.efficientnet_v1_aliases import (
    EfficientNetV1B7Backbone,
)
from keras_cv.models.backbones.efficientnet_v1.efficientnet_v1_aliases import (
    EfficientNetV1Backbone,
)
from keras_cv.models.backbones.efficientnet_v2.efficientnet_v2_aliases import (
    EfficientNetV2B0Backbone,
)
from keras_cv.models.backbones.efficientnet_v2.efficientnet_v2_aliases import (
    EfficientNetV2B1Backbone,
)
from keras_cv.models.backbones.efficientnet_v2.efficientnet_v2_aliases import (
    EfficientNetV2B2Backbone,
)
from keras_cv.models.backbones.efficientnet_v2.efficientnet_v2_aliases import (
    EfficientNetV2B3Backbone,
)
from keras_cv.models.backbones.efficientnet_v2.efficientnet_v2_aliases import (
    EfficientNetV2Backbone,
)
from keras_cv.models.backbones.efficientnet_v2.efficientnet_v2_aliases import (
    EfficientNetV2LBackbone,
)
from keras_cv.models.backbones.efficientnet_v2.efficientnet_v2_aliases import (
    EfficientNetV2MBackbone,
)
from keras_cv.models.backbones.efficientnet_v2.efficientnet_v2_aliases import (
    EfficientNetV2SBackbone,
)
from keras_cv.models.backbones.mix_transformer.mix_transformer_aliases import (
    MiTB0Backbone,
)
from keras_cv.models.backbones.mix_transformer.mix_transformer_aliases import (
    MiTB1Backbone,
)
from keras_cv.models.backbones.mix_transformer.mix_transformer_aliases import (
    MiTB2Backbone,
)
from keras_cv.models.backbones.mix_transformer.mix_transformer_aliases import (
    MiTB3Backbone,
)
from keras_cv.models.backbones.mix_transformer.mix_transformer_aliases import (
    MiTB4Backbone,
)
from keras_cv.models.backbones.mix_transformer.mix_transformer_aliases import (
    MiTB5Backbone,
)
from keras_cv.models.backbones.mix_transformer.mix_transformer_backbone import (
    MiTBackbone,
)
from keras_cv.models.backbones.mobilenet_v3.mobilenet_v3_aliases import (
    MobileNetV3LargeBackbone,
)
from keras_cv.models.backbones.mobilenet_v3.mobilenet_v3_aliases import (
    MobileNetV3SmallBackbone,
)
from keras_cv.models.backbones.mobilenet_v3.mobilenet_v3_backbone import (
    MobileNetV3Backbone,
)
from keras_cv.models.backbones.resnet_v1.resnet_v1_aliases import (
    ResNet18Backbone,
)
from keras_cv.models.backbones.resnet_v1.resnet_v1_aliases import (
    ResNet34Backbone,
)
from keras_cv.models.backbones.resnet_v1.resnet_v1_aliases import (
    ResNet50Backbone,
)
from keras_cv.models.backbones.resnet_v1.resnet_v1_aliases import (
    ResNet101Backbone,
)
from keras_cv.models.backbones.resnet_v1.resnet_v1_aliases import (
    ResNet152Backbone,
)
from keras_cv.models.backbones.resnet_v1.resnet_v1_backbone import (
    ResNetBackbone,
)
from keras_cv.models.backbones.resnet_v2.resnet_v2_aliases import (
    ResNet18V2Backbone,
)
from keras_cv.models.backbones.resnet_v2.resnet_v2_aliases import (
    ResNet34V2Backbone,
)
from keras_cv.models.backbones.resnet_v2.resnet_v2_aliases import (
    ResNet50V2Backbone,
)
from keras_cv.models.backbones.resnet_v2.resnet_v2_aliases import (
    ResNet101V2Backbone,
)
from keras_cv.models.backbones.resnet_v2.resnet_v2_aliases import (
    ResNet152V2Backbone,
)
from keras_cv.models.backbones.resnet_v2.resnet_v2_backbone import (
    ResNetV2Backbone,
)
from keras_cv.models.classification.image_classifier import ImageClassifier
from keras_cv.models.object_detection.retinanet.retinanet import RetinaNet
from keras_cv.models.object_detection.yolo_v8.yolo_v8_backbone import (
    YOLOV8Backbone,
)
from keras_cv.models.object_detection.yolo_v8.yolo_v8_detector import (
    YOLOV8Detector,
)
from keras_cv.models.segmentation import DeepLabV3Plus
from keras_cv.models.segmentation import SegFormer
from keras_cv.models.stable_diffusion import StableDiffusion
from keras_cv.models.stable_diffusion import StableDiffusionV2
