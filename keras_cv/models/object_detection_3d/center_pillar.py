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
from tensorflow import keras

from keras_cv.layers.object_detection_3d.heatmap_decoder import HeatmapDecoder


class MultiClassDetectionHead(keras.layers.Layer):
    """Multi-class object detection head."""

    def __init__(
        self,
        num_classes: int,
        num_head_bin: Sequence[int],
        share_head: bool = False,
        name: str = "detection_head",
    ):
        super().__init__(name=name)

        self._heads = {}
        self._head_names = []
        self._per_class_prediction_size = []
        self._num_classes = num_classes
        self._num_head_bin = num_head_bin
        for i in range(num_classes):
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

        if not share_head:
            for i in range(num_classes):
                # 1x1 conv for each voxel/pixel.
                self._heads[self._head_names[i]] = keras.layers.Conv2D(
                    filters=self._per_class_prediction_size[i],
                    kernel_size=(1, 1),
                    name=f"head_{i + 1}",
                )
        else:
            shared_layer = keras.layers.Conv2D(
                filters=self._per_class_prediction_size[0],
                kernel_size=(1, 1),
                name="shared_head",
            )
            for i in range(num_classes):
                self._heads[self._head_names[i]] = shared_layer

    def call(self, feature: tf.Tensor, training: bool) -> List[tf.Tensor]:
        del training
        outputs = {}
        for head_name in self._head_names:
            outputs[head_name] = self._heads[head_name](feature)
        return outputs


class MultiClassHeatmapDecoder(keras.layers.Layer):
    def __init__(
        self,
        num_classes,
        num_head_bin: Sequence[int],
        anchor_size: Sequence[Sequence[float]],
        max_pool_size: Sequence[int],
        max_num_box: Sequence[int],
        heatmap_threshold: Sequence[float],
        voxel_size: Sequence[float],
        spatial_size: Sequence[float],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.class_ids = list(range(1, num_classes + 1))
        self.num_head_bin = num_head_bin
        self.anchor_size = anchor_size
        self.max_pool_size = max_pool_size
        self.max_num_box = max_num_box
        self.heatmap_threshold = heatmap_threshold
        self.voxel_size = voxel_size
        self.spatial_size = spatial_size
        self.decoders = {}
        for i, class_id in enumerate(self.class_ids):
            self.decoders[f"class_{class_id}"] = HeatmapDecoder(
                class_id=class_id,
                num_head_bin=self.num_head_bin[i],
                anchor_size=self.anchor_size[i],
                max_pool_size=self.max_pool_size[i],
                max_num_box=self.max_num_box[i],
                heatmap_threshold=self.heatmap_threshold[i],
                voxel_size=self.voxel_size,
                spatial_size=self.spatial_size,
            )

    def call(self, predictions):
        box_predictions = []
        class_predictions = []
        box_confidence = []
        for k, v in predictions.items():
            boxes, classes, confidence = self.decoders[k](v)
            box_predictions.append(boxes)
            class_predictions.append(classes)
            box_confidence.append(confidence)

        return {
            "3d_boxes": {
                "boxes": tf.concat(box_predictions, axis=1),
                "classes": tf.concat(class_predictions, axis=1),
                "confidence": tf.concat(box_confidence, axis=1),
            }
        }


class MultiHeadCenterPillar(keras.Model):
    """Multi headed model based on CenterNet heatmap and PointPillar.

    This model builds box classification and regression for each class
    separately. It voxelizes the point cloud feature, applies feature extraction
    on top of voxelized feature, and applies multi-class classification and
    regression heads on the feature map.

    Args:
      backbone: the backbone to apply to voxelized features.
      voxel_net: the voxel_net that takes point cloud feature and convert
        to voxelized features.
      multiclass_head: a multi class head which returns a dict of heatmap prediction
        and regression prediction per class.
      label_encoder: a LabelEncoder that takes point cloud xyz and point cloud
        features and returns a multi class labels which is a dict of heatmap,
        box location and top_k heatmap index per class.
      prediction_decoder: a multi class heatmap prediction decoder that returns a dict
        of decoded boxes, box class, and box confidence score per class.


    """

    def __init__(
        self,
        backbone,
        voxel_net,
        multiclass_head,
        label_encoder,
        prediction_decoder,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._voxelization_layer = voxel_net
        self._unet_layer = backbone
        self._multiclass_head = multiclass_head
        self._label_encoder = label_encoder
        self._prediction_decoder = prediction_decoder
        self._head_names = self._multiclass_head._head_names

    def call(self, input_dict, training=None):
        point_xyz, point_feature, point_mask = (
            input_dict["point_xyz"],
            input_dict["point_feature"],
            input_dict["point_mask"],
        )
        voxel_feature = self._voxelization_layer(
            point_xyz, point_feature, point_mask, training=training
        )
        voxel_feature = self._unet_layer(voxel_feature, training=training)
        predictions = self._multiclass_head(voxel_feature)
        if not training:
            predictions = self._prediction_decoder(predictions)
        # returns dict {"class_1": concat_pred_1, "class_2": concat_pred_2}
        return predictions

    def compile(self, heatmap_loss=None, box_loss=None, **kwargs):
        """Compiles the MultiHeadCenterPillar.

        `compile()` mirrors the standard Keras `compile()` method, but allows
        for specification of heatmap and box-specific losses.

        Args:
            heatmap_loss: a Keras loss to use for heatmap regression.
            box_loss: a Keras loss to use for box regression.
            kwargs: other `keras.Model.compile()` arguments are supported and
                propagated to the `keras.Model` class.
        """
        losses = {}
        for i in range(self._multiclass_head._num_classes):
            losses[f"heatmap_class_{i+1}"] = heatmap_loss
            losses[f"box_class_{i+1}"] = box_loss

        super().compile(loss=losses, **kwargs)

    def compute_loss(self, predictions=None, targets=None):
        y_pred = {}
        y_true = {}
        sample_weight = {}
        for head_name in self._head_names:
            prediction = predictions[head_name]
            heatmap_pred = tf.nn.softmax(prediction[..., :2])[..., 1]
            box_pred = prediction[..., 2:]

            box = targets[head_name]["boxes"]
            heatmap = targets[head_name]["heatmap"]
            index = targets[head_name]["top_k_index"]

            # the prediction returns 2 outputs for background vs object
            y_pred["heatmap_" + head_name] = heatmap_pred
            y_true["heatmap_" + head_name] = heatmap
            sample_weight["heatmap_" + head_name] = tf.ones_like(heatmap)

            # TODO(ianstenbit): loss heatmap threshold should be configurable.
            box_regression_mask = (
                tf.gather_nd(heatmap, index, batch_dims=1) >= 0.95
            )
            sample_weight["box_" + head_name] = tf.cast(
                box_regression_mask, tf.float32
            )
            box = tf.gather_nd(box, index, batch_dims=1)
            box_pred = tf.gather_nd(box_pred, index, batch_dims=1)
            y_pred["box_" + head_name] = tf.squeeze(box_pred)
            y_true["box_" + head_name] = tf.squeeze(box)

        return super().compute_loss(
            x={}, y=y_true, y_pred=y_pred, sample_weight=sample_weight
        )

    def train_step(self, data):
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = self.compute_loss(predictions, y)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics({}, {}, {}, sample_weight={})
