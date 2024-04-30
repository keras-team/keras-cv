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
"""ImageClassifier Task presets."""


classifier_presets = {
    "resnet50_v2_imagenet_classifier": {
        "metadata": {
            "description": (
                "ResNet classifier with 50 layers where the batch "
                "normalization and ReLU activation precede the convolution "
                "layers (v2 style). Trained on Imagenet 2012 classification "
                "task."
            ),
            "params": 25_613_800,
            "official_name": "ImageClassifier",
            "path": "image_classifier",
        },
        "kaggle_handle": "kaggle://keras/resnetv2/keras/resnet50_v2_imagenet_classifier/2",  # noqa: E501
    },
    "efficientnetv2_s_imagenet_classifier": {
        "metadata": {
            "description": (
                "ImageClassifier using the EfficientNet small"
                "architecture.  In this "
                "variant of the EfficientNet architecture, there are "
                "6 convolutional blocks. Weights are "
                "initialized to pretrained imagenet classification weights."
                "Published weights are capable of scoring 83.9%	top 1 accuracy "
                "and 96.7% top 5 accuracy on imagenet."
            ),
            "params": 21_612_360,
            "official_name": "ImageClassifier",
            "path": "image_classifier",
        },
        "kaggle_handle": "kaggle://keras/efficientnetv2/keras/efficientnetv2_s_imagenet_classifier/2",  # noqa: E501
    },
    "efficientnetv2_b0_imagenet_classifier": {
        "metadata": {
            "description": (
                "ImageClassifier using the EfficientNet B0 "
                "architecture.  In this variant of the EfficientNet "
                "architecture, there are 6 convolutional blocks. As with all "
                "of the B style EfficientNet variants, the number of filters "
                "in each convolutional block is scaled by "
                "`width_coefficient=1.0` and "
                "`depth_coefficient=1.0`. Weights are "
                "initialized to pretrained imagenet classification weights. "
                "Published weights are capable of scoring 77.1%	top 1 accuracy "
                "and 93.3% top 5 accuracy on imagenet."
            ),
            "params": 7_200_312,
            "official_name": "ImageClassifier",
            "path": "image_classifier",
        },
        "kaggle_handle": "kaggle://keras/efficientnetv2/keras/efficientnetv2_b0_imagenet_classifier/2",  # noqa: E501
    },
    "efficientnetv2_b1_imagenet_classifier": {
        "metadata": {
            "description": (
                "ImageClassifier using the EfficientNet B1 "
                "architecture.  In this variant of the EfficientNet "
                "architecture, there are 6 convolutional blocks. As with all "
                "of the B style EfficientNet variants, the number of filters "
                "in each convolutional block is scaled by "
                "`width_coefficient=1.0` and "
                "`depth_coefficient=1.1`. Weights are "
                "initialized to pretrained imagenet classification weights."
                "Published weights are capable of scoring 79.1%	top 1 accuracy "
                "and 94.4% top 5 accuracy on imagenet."
            ),
            "params": 8_212_124,
            "official_name": "ImageClassifier",
            "path": "image_classifier",
        },
        "kaggle_handle": "kaggle://keras/efficientnetv2/keras/efficientnetv2_b1_imagenet_classifier/2",  # noqa: E501
    },
    "efficientnetv2_b2_imagenet_classifier": {
        "metadata": {
            "description": (
                "ImageClassifier using the EfficientNet B2 "
                "architecture.  In this variant of the EfficientNet "
                "architecture, there are 6 convolutional blocks. As with all "
                "of the B style EfficientNet variants, the number of filters "
                "in each convolutional block is scaled by "
                "`width_coefficient=1.1` and "
                "`depth_coefficient1.2`. Weights are initialized to pretrained "
                "imagenet classification weights."
                "Published weights are capable of scoring 80.1%	top 1 "
                "accuracy and 94.9% top 5 accuracy on imagenet."
            ),
            "params": 10_178_374,
            "official_name": "ImageClassifier",
            "path": "image_classifier",
        },
        "kaggle_handle": "kaggle://keras/efficientnetv2/keras/efficientnetv2_b2_imagenet_classifier/2",  # noqa: E501
    },
    "mobilenet_v3_large_imagenet_classifier": {
        "metadata": {
            "description": (
                "ImageClassifier using the MobileNetV3Large architecture. "
                "This preset uses a Dense layer as a classification head "
                "instead of the typical fully-convolutional MobileNet head. As "
                "a result, it has fewer parameters than the original "
                "MobileNetV3Large model, which has 5.4 million parameters."
                "Published weights are capable of scoring 69.4%	top-1 "
                "accuracy and 89.4% top 5 accuracy on imagenet."
            ),
            "params": 3_957_352,  # TODO this is wrong
            "official_name": "ImageClassifier",
            "path": "image_classifier",
        },
        "kaggle_handle": "kaggle://keras/mobilenetv3/keras/mobilenet_v3_large_imagenet_classifier/2",  # noqa: E501
    },
}
