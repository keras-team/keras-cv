# Copyright 2024 The KerasCV Authors
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
"""VideoClassifier Task presets."""

classifier_presets = {
    "videoswin_tiny_kinetics400_classifier": {
        "metadata": {
            "description": (
                "A tiny Video Swin architecture. "
                "It is pretrained on ImageNet 1K dataset, and "
                "trained on Kinetics 400 dataset. "
                "Published weight is capable of scoring "
                "78.8% top1 and 93.6% top5 accuracy on the "
                "Kinetics 400 dataset"
            ),
            "params": 28_158_070,
            "official_name": "VideoClassifier",
            "path": "video_classifier",
        },
    },
    "videoswin_small_kinetics400_classifier": {
        "metadata": {
            "description": (
                "A small Video Swin architecture. "
                "It is pretrained on ImageNet 1K dataset, and "
                "trained on Kinetics 400 dataset. "
                "Published weight is capable of scoring "
                "80.6% top1 and 94.5% top5 accuracy on the "
                "Kinetics 400 dataset"
            ),
            "params": 49_816_678,
            "official_name": "VideoClassifier",
            "path": "video_classifier",
        },
    },
    "videoswin_base_kinetics400_classifier": {
        "metadata": {
            "description": (
                "A base Video Swin architecture. "
                "It is pretrained on ImageNet 1K dataset, and "
                "trained on Kinetics 400 dataset. "
                "Published weight is capable of scoring "
                "80.6% top1 and 94.6% top5 accuracy on the "
                "Kinetics 400 dataset"
            ),
            "params": 89_065_688,
            "official_name": "VideoClassifier",
            "path": "video_classifier",
        },
    },
    "videoswin_base_kinetics400_imagenet22k": {
        "metadata": {
            "description": (
                "A base Video Swin architecture. "
                "It is pretrained on ImageNet 22K dataset, and "
                "trained on Kinetics 400 dataset. "
                "Published weight is capable of scoring "
                "82.7% top1 and 95.5% top5 accuracy on the "
                "Kinetics 400 dataset"
            ),
            "params": 89_065_688,
            "official_name": "VideoClassifier",
            "path": "video_classifier",
        },
    },
    "videoswin_base_kinetics600_imagenet22k": {
        "metadata": {
            "description": (
                "A base Video Swin architecture. "
                "It is pretrained on ImageNet 22K dataset, and "
                "trained on Kinetics 600 dataset. "
                "Published weight is capable of scoring "
                "84.0% top1 and 96.5% top5 accuracy on the "
                "Kinetics 600 dataset"
            ),
            "params": 89_270_688,
            "official_name": "VideoClassifier",
            "path": "video_classifier",
        },
    },
    "videoswin_base_something_something_v2_classifier": {
        "metadata": {
            "description": (
                "A base Video Swin architecture. "
                "It is pretrained on Kinetics 400 dataset, and "
                "trained on Something Something V2 dataset. "
                "Published weight is capable of scoring "
                "69.6% top1 and 92.7% top5 accuracy on the "
                "Kinetics 400 dataset"
            ),
            "params": 88_834_038,
            "official_name": "VideoClassifier",
            "path": "video_classifier",
        },
    },
}
