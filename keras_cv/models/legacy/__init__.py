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

from keras_cv.models.legacy.convmixer import ConvMixer_512_16
from keras_cv.models.legacy.convmixer import ConvMixer_768_32
from keras_cv.models.legacy.convmixer import ConvMixer_1024_16
from keras_cv.models.legacy.convmixer import ConvMixer_1536_20
from keras_cv.models.legacy.convmixer import ConvMixer_1536_24
from keras_cv.models.legacy.darknet import DarkNet21
from keras_cv.models.legacy.darknet import DarkNet53
from keras_cv.models.legacy.densenet import DenseNet121
from keras_cv.models.legacy.densenet import DenseNet169
from keras_cv.models.legacy.densenet import DenseNet201
from keras_cv.models.legacy.efficientnet_lite import EfficientNetLiteB0
from keras_cv.models.legacy.efficientnet_lite import EfficientNetLiteB1
from keras_cv.models.legacy.efficientnet_lite import EfficientNetLiteB2
from keras_cv.models.legacy.efficientnet_lite import EfficientNetLiteB3
from keras_cv.models.legacy.efficientnet_lite import EfficientNetLiteB4
from keras_cv.models.legacy.efficientnet_v1 import EfficientNetB0
from keras_cv.models.legacy.efficientnet_v1 import EfficientNetB1
from keras_cv.models.legacy.efficientnet_v1 import EfficientNetB2
from keras_cv.models.legacy.efficientnet_v1 import EfficientNetB3
from keras_cv.models.legacy.efficientnet_v1 import EfficientNetB4
from keras_cv.models.legacy.efficientnet_v1 import EfficientNetB5
from keras_cv.models.legacy.efficientnet_v1 import EfficientNetB6
from keras_cv.models.legacy.efficientnet_v1 import EfficientNetB7
from keras_cv.models.legacy.mlp_mixer import MLPMixerB16
from keras_cv.models.legacy.mlp_mixer import MLPMixerB32
from keras_cv.models.legacy.mlp_mixer import MLPMixerL16
from keras_cv.models.legacy.object_detection.faster_rcnn.faster_rcnn import (
    FasterRCNN,
)
from keras_cv.models.legacy.regnet import RegNetX002
from keras_cv.models.legacy.regnet import RegNetX004
from keras_cv.models.legacy.regnet import RegNetX006
from keras_cv.models.legacy.regnet import RegNetX008
from keras_cv.models.legacy.regnet import RegNetX016
from keras_cv.models.legacy.regnet import RegNetX032
from keras_cv.models.legacy.regnet import RegNetX040
from keras_cv.models.legacy.regnet import RegNetX064
from keras_cv.models.legacy.regnet import RegNetX080
from keras_cv.models.legacy.regnet import RegNetX120
from keras_cv.models.legacy.regnet import RegNetX160
from keras_cv.models.legacy.regnet import RegNetX320
from keras_cv.models.legacy.regnet import RegNetY002
from keras_cv.models.legacy.regnet import RegNetY004
from keras_cv.models.legacy.regnet import RegNetY006
from keras_cv.models.legacy.regnet import RegNetY008
from keras_cv.models.legacy.regnet import RegNetY016
from keras_cv.models.legacy.regnet import RegNetY032
from keras_cv.models.legacy.regnet import RegNetY040
from keras_cv.models.legacy.regnet import RegNetY064
from keras_cv.models.legacy.regnet import RegNetY080
from keras_cv.models.legacy.regnet import RegNetY120
from keras_cv.models.legacy.regnet import RegNetY160
from keras_cv.models.legacy.regnet import RegNetY320
from keras_cv.models.legacy.segmentation.deeplab import DeepLabV3
from keras_cv.models.legacy.vgg16 import VGG16
from keras_cv.models.legacy.vgg19 import VGG19
from keras_cv.models.legacy.vit import ViTB16
from keras_cv.models.legacy.vit import ViTB32
from keras_cv.models.legacy.vit import ViTH16
from keras_cv.models.legacy.vit import ViTH32
from keras_cv.models.legacy.vit import ViTL16
from keras_cv.models.legacy.vit import ViTL32
from keras_cv.models.legacy.vit import ViTS16
from keras_cv.models.legacy.vit import ViTS32
from keras_cv.models.legacy.vit import ViTTiny16
from keras_cv.models.legacy.vit import ViTTiny32
