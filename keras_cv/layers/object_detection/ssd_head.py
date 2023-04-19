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


import tensorflow as tf
from typing import Union


class SSDHead(tf.keras.layers.Layer):
    """
    Implementation of head module of SSD: Single Shot MultiBox Detector

    Reference:
    [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
    [SSD: Torchvision](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py)

    Arguments:
        num_anchors: tf.Tensor - Tensor contains number of anchors at different scales
        num_classes: int - Integer represents total number of classes considered for classification

    Returns:
        A Python dictionary with the following format,
        {
            "classification_results": tf.Tensor with shape(num_anchors, params, num_classes)
            "bbox_regression_results": tf.Tensor with shape (num_anchors, params, 4)
        }
    """
    def __init__(self,
                 num_anchors: tf.Tensor,
                 num_classes: int,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.classification_head = self.conv_ssd_head(self.num_anchors,
                                                      self.num_classes,
                                                      name="cls")
        self.regression_head = self.conv_ssd_head(self.num_anchors,
                                                  4,
                                                  name="reg")

    def conv_ssd_head(self,
                      num_anchors: tf.Tensor,
                      num_classes: int,
                      name: str):
        """
        Creates a list of Conv2D layers to construct classification or regression block of head

        Returns:
            A Python List of tf.keras.layers.Conv2D objects
        """
        block = list()
        for idx, anchor in enumerate(num_anchors):
            block.append(
                tf.keras.layers.Conv2D(filters=num_classes * anchor,
                                       kernel_size=3,
                                       padding="same",
                                       name=f"{name}_{idx}")
            )
        return block

    def reshape_features(self,
                         x: tf.Tensor,
                         num_columns: int):
        """
        Reduce the output from Conv2D layer into 2D tensor

        Returns:
            A tf.Tensor
        """
        N, H, W, C = x.shape
        if N is None:
            x = tf.reshape(x, shape=(1, H, W, C))
            N = 1
        x = tf.reshape(x, shape=(N, H, W, -1, num_columns))
        x = tf.reshape(x, shape=(N, -1, num_columns))
        return x

    def _check_tensor(self,
                      x: Union[tf.Tensor, list[tf.Tensor]]):
        """
        Function to test the input Tensor or List of features

        Returns:
            None
        """
        if type(x) == list:
            for feature in x:
                assert len(feature.shape) == 4, \
                    f"The input list should contain Tensor with total dimensions of 4 but got {len(feature.shape)}"

        else:
            assert len(x.shape) == 5, \
                "The input tensor should have 5 dimensions, (num_scales, N, H, W, C)"
            assert x.shape[0] == len(self.num_anchors), \
                "The first dimension of input should be equal to number of anchors"

    def call(self,
             x: Union[tf.Tensor, list[tf.Tensor]]):
        self._check_tensor(x)

        classification_results = list()
        bbox_results = list()

        for i, features in enumerate(x):
            cls_result = self.classification_head[i](features)
            bbox_result = self.regression_head[i](features)

            cls_result = self.reshape_features(cls_result,
                                               self.num_classes)
            bbox_result = self.reshape_features(bbox_result, 4)

            classification_results.append(cls_result)
            bbox_results.append(bbox_result)

        classification_results = tf.concat(classification_results,
                                           axis=1)
        bbox_results = tf.concat(bbox_results,
                                 axis=1)

        return {"classification_results": classification_results,
                "bbox_regression_results": bbox_results}
