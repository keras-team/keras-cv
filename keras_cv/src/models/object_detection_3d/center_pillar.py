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

import copy

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.layers.object_detection_3d.heatmap_decoder import (
    HeatmapDecoder,
)
from keras_cv.src.models.object_detection_3d.center_pillar_backbone_presets import (  # noqa: E501
    backbone_presets,
)
from keras_cv.src.models.task import Task
from keras_cv.src.utils.python_utils import classproperty


@keras_cv_export("keras_cv.models.MultiHeadCenterPillar")
class MultiHeadCenterPillar(Task):
    """Multi headed model based on CenterNet heatmap and PointPillar.

    This model builds box classification and regression for each class
    separately. It voxelizes the point cloud feature, applies feature extraction
    on top of voxelized feature, and applies multi-class classification and
    regression heads on the feature map.

    Args:
        backbone: the backbone to apply to voxelized features.
        voxel_net: the voxel_net that takes point cloud feature and convert
            to voxelized features. KerasCV offers a `DynamicVoxelization` layer
            in `keras_cv.layers` which is a reasonable default for most
            detection use cases.
        multiclass_head: A keras.layers.Layer which takes the backbone output
            and returns a dict of heatmap prediction and regression prediction
            per class.
        prediction_decoder: a multi class heatmap prediction decoder that
            returns a dict of decoded boxes, box class, and box confidence score
            per class.
    """

    def __init__(
        self,
        backbone,
        voxel_net,
        multiclass_head,
        prediction_decoder,
        **kwargs,
    ):
        point_xyz = keras.layers.Input((None, 3), name="point_xyz")
        point_feature = keras.layers.Input((None, 4), name="point_feature")
        point_mask = keras.layers.Input(
            (None, 1), name="point_mask", dtype="bool"
        )

        inputs = {
            "point_xyz": point_xyz,
            "point_feature": point_feature,
            "point_mask": point_mask,
        }

        voxel_feature = voxel_net(point_xyz, point_feature, point_mask[..., 0])
        voxel_feature = backbone(voxel_feature)
        predictions = multiclass_head(voxel_feature)

        # A slight hack to get the output names in the model outputs for a
        # functional model.
        for head_name in multiclass_head._head_names:
            predictions[f"box_{head_name}"] = keras.layers.Identity(
                name=f"box_{head_name}"
            )(predictions[head_name])
            predictions[f"heatmap_{head_name}"] = keras.layers.Identity(
                name=f"heatmap_{head_name}"
            )(predictions[head_name])

        super().__init__(inputs=inputs, outputs=predictions, **kwargs)
        self._backbone = backbone
        self._multiclass_head = multiclass_head
        self._prediction_decoder = prediction_decoder
        self._head_names = self._multiclass_head._head_names

    def compile(self, heatmap_loss=None, box_loss=None, **kwargs):
        """Compiles the MultiHeadCenterPillar.

        `compile()` mirrors the standard Keras `compile()` method, but allows
        for specification of heatmap and box-specific losses.

        Args:
            heatmap_loss: a Keras loss to use for heatmap regression.
            box_loss: a Keras loss to use for box regression, or a list of Keras
                losses for box regression, one for each class. If only one loss
                is specified, it will be used for all classes, otherwise exactly
                one loss should be specified per class.
            kwargs: other `keras.Model.compile()` arguments are supported and
                propagated to the `keras.Model` class.
        """
        losses = {}

        if box_loss is not None and not isinstance(box_loss, list):
            box_loss = [
                box_loss for _ in range(self._multiclass_head._num_classes)
            ]
        for i in range(self._multiclass_head._num_classes):
            losses[f"heatmap_class_{i+1}"] = heatmap_loss
            losses[f"box_class_{i+1}"] = box_loss[i]

        super().compile(loss=losses, **kwargs)

    def compute_loss(self, x, y, y_pred, sample_weight=None, **kwargs):
        predictions = y_pred
        targets = y

        y_pred = {}
        y_true = {}
        sample_weight = {}

        for head_name in self._head_names:
            prediction = predictions[head_name]
            heatmap_pred = ops.softmax(prediction[..., :2])[..., 1]
            box_pred = prediction[..., 2:]
            box = targets[head_name]["boxes"]
            heatmap = targets[head_name]["heatmap"]
            index = targets[head_name]["top_k_index"]

            # the prediction returns 2 outputs for background vs object
            y_pred["heatmap_" + head_name] = heatmap_pred
            y_true["heatmap_" + head_name] = heatmap

            # TODO(ianstenbit): loss heatmap threshold should be configurable.
            box_regression_mask = (
                ops.take_along_axis(
                    ops.reshape(heatmap, (heatmap.shape[0], -1)),
                    index[..., 0] * heatmap.shape[1] + index[..., 1],
                    axis=1,
                )
                > 0.95
            )

            box = ops.take_along_axis(
                ops.reshape(box, (ops.shape(box)[0], -1, 7)),
                ops.expand_dims(
                    index[..., 0] * ops.shape(box)[1] + index[..., 1], axis=-1
                ),
                axis=1,
            )

            box_pred = ops.take_along_axis(
                ops.reshape(
                    box_pred,
                    (ops.shape(box_pred)[0], -1, ops.shape(box_pred)[-1]),
                ),
                ops.expand_dims(
                    index[..., 0] * ops.shape(box_pred)[1] + index[..., 1],
                    axis=-1,
                ),
                axis=1,
            )

            box_center_mask = heatmap > 0.99
            num_boxes = ops.maximum(
                ops.sum(ops.cast(box_center_mask, "float32"), axis=[1, 2]), 1
            )

            sample_weight["box_" + head_name] = ops.cast(
                box_regression_mask, "float32"
            ) / ops.broadcast_to(
                ops.expand_dims(num_boxes, axis=-1),
                ops.shape(box_regression_mask),
            )
            sample_weight["heatmap_" + head_name] = ops.ones_like(
                heatmap
            ) / ops.broadcast_to(
                ops.expand_dims(ops.expand_dims(num_boxes, axis=-1), axis=-1),
                heatmap.shape,
            )

            y_pred["box_" + head_name] = box_pred
            y_true["box_" + head_name] = box

        return super().compute_loss(
            x={}, y=y_true, y_pred=y_pred, sample_weight=sample_weight
        )

    def predict_step(self, *args):
        outputs = super().predict_step(*args)
        if isinstance(outputs, tuple):
            return self._prediction_decoder(outputs[0]), outputs[1]
        else:
            return self._prediction_decoder(outputs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return copy.deepcopy(backbone_presets)

    @classproperty
    def backbone_presets(cls):
        """Dictionary of preset names and configurations of compatible
        backbones."""
        return copy.deepcopy(backbone_presets)


class MultiClassDetectionHead(keras.layers.Layer):
    """Multi-class object detection head for CenterPillar.

    This head includes a 1x1 convolution layer for each class which is called
    on the output of the CenterPillar's backbone. The outputs are per-class
    prediction heatmaps which must be decoded into 3D boxes.

    Args:
        num_classes: int, the number of box classes to predict.
        num_head_bin: list of ints, the number of heading bins to use for each
            respective box class.
    """

    def __init__(
        self,
        num_classes,
        num_head_bin,
        name="detection_head",
    ):
        super().__init__(name=name)

        self._heads = {}
        self._head_names = []
        self._num_classes = num_classes
        self._num_head_bin = num_head_bin

        for i in range(num_classes):
            self._head_names.append(f"class_{i + 1}")

            # 1x1 conv for each voxel/pixel.
            self._heads[self._head_names[i]] = keras.layers.Conv2D(
                # 2 for class, 3 for location, 3 for size, 2N for heading
                filters=8 + 2 * num_head_bin[i],
                kernel_size=(1, 1),
                name=f"head_{i + 1}",
            )

    def call(self, feature, training=True):
        del training
        outputs = {}
        for head_name in self._head_names:
            outputs[head_name] = self._heads[head_name](feature)
        return outputs


class MultiClassHeatmapDecoder(keras.layers.Layer):
    """Heatmap decoder for CenterPillar models.

    The heatmap decoder converts a sparse heatmap of box predictions into a
    padded dense set of decoded predicted boxes.

    The input to the heatmap decoder is a spatial heatmap of encoded box
    predictions, and the output is decoded 3D boxes in CENTER_XYZ_DXDYDZ_PHI
    format.

    Args:
        num_classes: int, the number of box classes to predict.
        num_head_bin: list of ints, the number of heading bins for each
            respective class.
        anchor_size: list of length-3 lists of floats, the 3D anchor sizes for
            each respective class.
        max_pool_size: list of ints, the 2D pooling size for the heatmap, to be
            used before box decoding.
        max_num_box: list of ints, the maximum number of boxes to return for
            each class. The top K boxes will be returned, and if fewer than K
            boxes are predicted, the outputs will be padded to contain K boxes.
        heatmap_threshold: list of floats, the heatmap confidence threshold to
            be used for each respective class to determine whether or not a box
            prediction is strong enough to decode and return.
        voxel_size: list of floats, the size of the voxels that were used to
            voxelize inputs to the CenterPillar model for each respective class.
        spatial_size: list of floats, the global 3D size of the heatmap for each
            respective class. `spatial_size[i] / voxel_size[i]` equals the
            size of the `i`th rank of the input heatmap.
    """

    def __init__(
        self,
        num_classes,
        num_head_bin,
        anchor_size,
        max_pool_size,
        max_num_box,
        heatmap_threshold,
        voxel_size,
        spatial_size,
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
        for class_id in self.class_ids:
            class_tag = f"class_{class_id}"
            boxes, classes, confidence = self.decoders[class_tag](
                predictions[class_tag]
            )
            box_predictions.append(boxes)
            class_predictions.append(classes)
            box_confidence.append(confidence)

        return {
            "3d_boxes": {
                "boxes": ops.concatenate(box_predictions, axis=1),
                "classes": ops.concatenate(class_predictions, axis=1),
                "confidence": ops.concatenate(box_confidence, axis=1),
            }
        }
