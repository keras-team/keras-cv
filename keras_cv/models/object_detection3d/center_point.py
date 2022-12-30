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

from typing import List
from typing import Sequence

import tensorflow as tf

from keras_cv.models.__internal__.unet import UNet

down_block_configs = [(128, 6), (256, 2), (512, 2)]
up_block_configs = [512, 256, 256]


class MultiClassDetectionHead(tf.keras.layers.Layer):
    """Multi-class object detection head."""

    def __init__(
        self,
        num_class: int,
        num_head_bin: Sequence[int],
        share_head: bool = False,
        name: str = "detection_head",
    ):
        super().__init__(name=name)

        self._heads = []
        self._head_names = []
        self._prediction_names = ["heatmap", "offset", "dimension", "heading"]
        self._per_class_prediction_size = []
        self._num_class = num_class
        self._num_head_bin = num_head_bin
        for i in range(num_class):
            self._head_names.append(f"class_{i + 1}")
            size = 0
            # 0:1 outputs is for classification
            size += 2
            # 2:4 outputs is for location offset
            size += 3
            # 5:7 outputs is for dimension offset
            size += 3
            # 8:end outputs is for bin-based classification and regression
            size += 2 * num_head_bin[i]
            self._per_class_prediction_size.append(size)
        print("per class prediction size {}".format(self._per_class_prediction_size))
        if not share_head:
            for i in range(num_class):
                # 1x1 conv for each voxel/pixel.
                self._heads[self._head_names[i]] = tf.keras.layers.Conv2D(
                    filters=self._per_class_prediction_size[i],
                    kernel_size=(1, 1),
                    name=f"head_{i + 1}",
                )
        else:
            shared_layer = tf.keras.layers.Conv2D(
                filters=self._per_class_prediction_size[0],
                kernel_size=(1, 1),
                name="shared_head",
            )
            for i in range(num_class):
                self._heads[self._head_names[i]] = shared_layer

    def call(self, feature: tf.Tensor, training: bool) -> List[tf.Tensor]:
        del training
        outputs = {}
        for head_name in self._head_names:
            outputs[head_name] = self._heads[head_name](feature)
        return outputs


class MultiHeadCenterPoint(tf.keras.Model):
    def __init__(
        self,
        backbone,
        voxel_net,
        multiclass_head,
        label_encoder=None,
        prediction_decoder=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._voxelization_layer = voxel_net
        self._unet_layer = backbone or UNet(down_block_configs, up_block_configs)
        self._multiclass_head = multiclass_head or MultiClassDetectionHead(2, [12, 4])
        self._prediction_decoder = prediction_decoder
        self._head_names = self._multiclass_head._head_names

    def call(self, point_xyz, point_feature, point_mask, training=None):
        predictions = self._forward(
            point_xyz, point_feature, point_mask, training=training
        )
        voxel_feature = self._voxelization_layer(
            point_xyz, point_feature, point_mask, training=training
        )
        voxel_feature = self._unet_layer(voxel_feature, training=training)
        predictions = self._multiclass_head(voxel_feature)
        # if not training:
        #     predictions = self._prediction_decoder(predictions)
        # returns {"class_1": concat_pred_1, "class_2": concat_pred_2}
        return predictions

    def compute_loss(self, predictions, box_dict, heatmap_dict, top_k_index_dict):
        y_pred = {}
        y_true = {}
        sample_weight = {}
        for head_name in self._head_names:
            prediction = predictions[head_name]
            heatmap_pred = tf.nn.softmax(prediction[..., :2])[..., 1]
            box_pred = prediction[..., 2:]
            box = box_dict[head_name]
            heatmap = heatmap_dict[head_name]
            index = top_k_index_dict[head_name]
            # the prediction returns 2 outputs for background vs object
            y_pred["heatmap_" + head_name] = heatmap_pred
            y_true["heatmap_" + head_name] = heatmap
            sample_weight["heatmap_" + head_name] = tf.ones_like(heatmap)
            # heatmap_groundtruth_gather = tf.gather_nd(heatmap, index, batch_dims=1)
            # TODO(tanzhenyu): loss heatmap threshold be configurable.
            # box_regression_mask = heatmap_groundtruth_gather >= 0.95
            box = tf.gather_nd(box, index, batch_dims=1)
            box_pred = tf.gather_nd(box_pred, index, batch_dims=1)
            y_pred["bin_" + head_name] = box_pred
            y_true["bin_" + head_name] = box

        return super().compute_loss(
            x={}, y=y_true, y_pred=y_pred, sample_weight=sample_weight
        )

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        box_3d_dict = y["box_3d"]
        heatmap_dict = y["heatmap"]
        top_k_index_dict = y["top_k_index"]
        losses = []
        with tf.GradientTape() as tape:
            predictions = self(
                x["point_xyz"], x["point_feature"], x["point_mask"], training=True
            )
            losses.append(
                self.compute_loss(
                    predictions, box_3d_dict, heatmap_dict, top_k_index_dict
                )
            )
            if self.weight_decay:
                for var in self.trainable_variables:
                    losses.append(self.weight_decay * tf.nn.l2_loss(var))
            total_loss = tf.math.add_n(losses)
        self.optimizer.minimize(total_loss, self.trainable_variables, tape=tape)
        return self.compute_metrics({}, {}, {}, sample_weight={})
