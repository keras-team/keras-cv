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

from keras_cv.src.models import legacy
from keras_cv.src.models.backbones.csp_darknet.csp_darknet_aliases import (
    CSPDarkNetLBackbone,
)
from keras_cv.src.models.backbones.csp_darknet.csp_darknet_aliases import (
    CSPDarkNetMBackbone,
)
from keras_cv.src.models.backbones.csp_darknet.csp_darknet_aliases import (
    CSPDarkNetSBackbone,
)
from keras_cv.src.models.backbones.csp_darknet.csp_darknet_aliases import (
    CSPDarkNetTinyBackbone,
)
from keras_cv.src.models.backbones.csp_darknet.csp_darknet_aliases import (
    CSPDarkNetXLBackbone,
)
from keras_cv.src.models.backbones.csp_darknet.csp_darknet_backbone import (
    CSPDarkNetBackbone,
)
from keras_cv.src.models.backbones.densenet.densenet_aliases import (
    DenseNet121Backbone,
)
from keras_cv.src.models.backbones.densenet.densenet_aliases import (
    DenseNet169Backbone,
)
from keras_cv.src.models.backbones.densenet.densenet_aliases import (
    DenseNet201Backbone,
)
from keras_cv.src.models.backbones.densenet.densenet_backbone import (
    DenseNetBackbone,
)
from keras_cv.src.models.backbones.efficientnet_lite.efficientnet_lite_aliases import (  # noqa: E501
    EfficientNetLiteB0Backbone,
)
from keras_cv.src.models.backbones.efficientnet_lite.efficientnet_lite_aliases import (  # noqa: E501
    EfficientNetLiteB1Backbone,
)
from keras_cv.src.models.backbones.efficientnet_lite.efficientnet_lite_aliases import (  # noqa: E501
    EfficientNetLiteB2Backbone,
)
from keras_cv.src.models.backbones.efficientnet_lite.efficientnet_lite_aliases import (  # noqa: E501
    EfficientNetLiteB3Backbone,
)
from keras_cv.src.models.backbones.efficientnet_lite.efficientnet_lite_aliases import (  # noqa: E501
    EfficientNetLiteB4Backbone,
)
from keras_cv.src.models.backbones.efficientnet_lite.efficientnet_lite_backbone import (  # noqa: E501
    EfficientNetLiteBackbone,
)
from keras_cv.src.models.backbones.efficientnet_v1.efficientnet_v1_aliases import (  # noqa: E501
    EfficientNetV1B0Backbone,
)
from keras_cv.src.models.backbones.efficientnet_v1.efficientnet_v1_aliases import (  # noqa: E501
    EfficientNetV1B1Backbone,
)
from keras_cv.src.models.backbones.efficientnet_v1.efficientnet_v1_aliases import (  # noqa: E501
    EfficientNetV1B2Backbone,
)
from keras_cv.src.models.backbones.efficientnet_v1.efficientnet_v1_aliases import (  # noqa: E501
    EfficientNetV1B3Backbone,
)
from keras_cv.src.models.backbones.efficientnet_v1.efficientnet_v1_aliases import (  # noqa: E501
    EfficientNetV1B4Backbone,
)
from keras_cv.src.models.backbones.efficientnet_v1.efficientnet_v1_aliases import (  # noqa: E501
    EfficientNetV1B5Backbone,
)
from keras_cv.src.models.backbones.efficientnet_v1.efficientnet_v1_aliases import (  # noqa: E501
    EfficientNetV1B6Backbone,
)
from keras_cv.src.models.backbones.efficientnet_v1.efficientnet_v1_aliases import (  # noqa: E501
    EfficientNetV1B7Backbone,
)
from keras_cv.src.models.backbones.efficientnet_v1.efficientnet_v1_aliases import (  # noqa: E501
    EfficientNetV1Backbone,
)
from keras_cv.src.models.backbones.efficientnet_v2.efficientnet_v2_aliases import (  # noqa: E501
    EfficientNetV2B0Backbone,
)
from keras_cv.src.models.backbones.efficientnet_v2.efficientnet_v2_aliases import (  # noqa: E501
    EfficientNetV2B1Backbone,
)
from keras_cv.src.models.backbones.efficientnet_v2.efficientnet_v2_aliases import (  # noqa: E501
    EfficientNetV2B2Backbone,
)
from keras_cv.src.models.backbones.efficientnet_v2.efficientnet_v2_aliases import (  # noqa: E501
    EfficientNetV2B3Backbone,
)
from keras_cv.src.models.backbones.efficientnet_v2.efficientnet_v2_aliases import (  # noqa: E501
    EfficientNetV2Backbone,
)
from keras_cv.src.models.backbones.efficientnet_v2.efficientnet_v2_aliases import (  # noqa: E501
    EfficientNetV2LBackbone,
)
from keras_cv.src.models.backbones.efficientnet_v2.efficientnet_v2_aliases import (  # noqa: E501
    EfficientNetV2MBackbone,
)
from keras_cv.src.models.backbones.efficientnet_v2.efficientnet_v2_aliases import (  # noqa: E501
    EfficientNetV2SBackbone,
)
from keras_cv.src.models.backbones.mix_transformer.mix_transformer_aliases import (  # noqa: E501
    MiTB0Backbone,
)
from keras_cv.src.models.backbones.mix_transformer.mix_transformer_aliases import (  # noqa: E501
    MiTB1Backbone,
)
from keras_cv.src.models.backbones.mix_transformer.mix_transformer_aliases import (  # noqa: E501
    MiTB2Backbone,
)
from keras_cv.src.models.backbones.mix_transformer.mix_transformer_aliases import (  # noqa: E501
    MiTB3Backbone,
)
from keras_cv.src.models.backbones.mix_transformer.mix_transformer_aliases import (  # noqa: E501
    MiTB4Backbone,
)
from keras_cv.src.models.backbones.mix_transformer.mix_transformer_aliases import (  # noqa: E501
    MiTB5Backbone,
)
from keras_cv.src.models.backbones.mix_transformer.mix_transformer_aliases import (  # noqa: E501
    MiTBackbone,
)
from keras_cv.src.models.backbones.mobilenet_v3.mobilenet_v3_aliases import (
    MobileNetV3LargeBackbone,
)
from keras_cv.src.models.backbones.mobilenet_v3.mobilenet_v3_aliases import (
    MobileNetV3SmallBackbone,
)
from keras_cv.src.models.backbones.mobilenet_v3.mobilenet_v3_backbone import (
    MobileNetV3Backbone,
)
from keras_cv.src.models.backbones.resnet_v1.resnet_v1_aliases import (
    ResNet18Backbone,
)
from keras_cv.src.models.backbones.resnet_v1.resnet_v1_aliases import (
    ResNet34Backbone,
)
from keras_cv.src.models.backbones.resnet_v1.resnet_v1_aliases import (
    ResNet50Backbone,
)
from keras_cv.src.models.backbones.resnet_v1.resnet_v1_aliases import (
    ResNet101Backbone,
)
from keras_cv.src.models.backbones.resnet_v1.resnet_v1_aliases import (
    ResNet152Backbone,
)
from keras_cv.src.models.backbones.resnet_v1.resnet_v1_backbone import (
    ResNetBackbone,
)
from keras_cv.src.models.backbones.resnet_v2.resnet_v2_aliases import (
    ResNet18V2Backbone,
)
from keras_cv.src.models.backbones.resnet_v2.resnet_v2_aliases import (
    ResNet34V2Backbone,
)
from keras_cv.src.models.backbones.resnet_v2.resnet_v2_aliases import (
    ResNet50V2Backbone,
)
from keras_cv.src.models.backbones.resnet_v2.resnet_v2_aliases import (
    ResNet101V2Backbone,
)
from keras_cv.src.models.backbones.resnet_v2.resnet_v2_aliases import (
    ResNet152V2Backbone,
)
from keras_cv.src.models.backbones.resnet_v2.resnet_v2_backbone import (
    ResNetV2Backbone,
)
from keras_cv.src.models.backbones.vgg16.vgg16_backbone import VGG16Backbone
from keras_cv.src.models.backbones.video_swin.video_swin_aliases import (
    VideoSwinBBackbone,
)
from keras_cv.src.models.backbones.video_swin.video_swin_aliases import (
    VideoSwinSBackbone,
)
from keras_cv.src.models.backbones.video_swin.video_swin_aliases import (
    VideoSwinTBackbone,
)
from keras_cv.src.models.backbones.video_swin.video_swin_backbone import (
    VideoSwinBackbone,
)
from keras_cv.src.models.backbones.vit_det.vit_det_aliases import (
    ViTDetBBackbone,
)
from keras_cv.src.models.backbones.vit_det.vit_det_aliases import (
    ViTDetHBackbone,
)
from keras_cv.src.models.backbones.vit_det.vit_det_aliases import (
    ViTDetLBackbone,
)
from keras_cv.src.models.backbones.vit_det.vit_det_backbone import (
    ViTDetBackbone,
)
from keras_cv.src.models.classification.image_classifier import ImageClassifier
from keras_cv.src.models.classification.video_classifier import VideoClassifier
from keras_cv.src.models.feature_extractor.clip import CLIP
from keras_cv.src.models.object_detection.retinanet.retinanet import RetinaNet
from keras_cv.src.models.object_detection.yolo_v8.yolo_v8_backbone import (
    YOLOV8Backbone,
)
from keras_cv.src.models.object_detection.yolo_v8.yolo_v8_detector import (
    YOLOV8Detector,
)
from keras_cv.src.models.segmentation import BASNet
from keras_cv.src.models.segmentation import DeepLabV3Plus
from keras_cv.src.models.segmentation import SAMMaskDecoder
from keras_cv.src.models.segmentation import SAMPromptEncoder
from keras_cv.src.models.segmentation import SegmentAnythingModel
from keras_cv.src.models.segmentation import TwoWayTransformer
from keras_cv.src.models.segmentation.segformer.segformer_aliases import (
    SegFormer,
)
from keras_cv.src.models.segmentation.segformer.segformer_aliases import (
    SegFormerB0,
)
from keras_cv.src.models.segmentation.segformer.segformer_aliases import (
    SegFormerB1,
)
from keras_cv.src.models.segmentation.segformer.segformer_aliases import (
    SegFormerB2,
)
from keras_cv.src.models.segmentation.segformer.segformer_aliases import (
    SegFormerB3,
)
from keras_cv.src.models.segmentation.segformer.segformer_aliases import (
    SegFormerB4,
)
from keras_cv.src.models.segmentation.segformer.segformer_aliases import (
    SegFormerB5,
)
from keras_cv.src.models.stable_diffusion import StableDiffusion
from keras_cv.src.models.stable_diffusion import StableDiffusionV2
