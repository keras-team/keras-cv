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

import numpy as np

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.layers.object_detection_3d import voxel_utils


def decode_bin_heading(predictions, num_bin):
    """Decode bin heading.

    Computes the box heading (orientation) by decoding the bin predictions. The
    predictions should contain bin classification scores (first num_bin scores)
    and corresponding bin residuals (the following num_bin scores).

    Args:
      predictions: Prediction scores tensor with size [N, num_bin*2]
        predictions = [:, bin_1, bin_2, ..., bin_k, res_1, res_2, ..., res_k],
        where k is the number of bins and N is the number of boxes.
      num_bin: A constant showing the number of bins used in heading bin loss.

    Returns:
      heading: Decoded heading tensor with size [N] in which heading values are
        in the [-pi, pi] range.

    Raises:
      ValueError: If the rank of `predictions` is not 2 or `predictions` tensor
        does not more than the expected number of dimensions.
    """
    with keras.backend.name_scope("decode_bin_heading"):
        if len(predictions.shape) != 2:
            raise ValueError(
                "The rank of the prediction tensor is expected to be 2. "
                f"Instead it is : {len(predictions.shape)}."
            )

        # Get the index of the bin with the maximum score to build a tensor of
        # [N].
        bin_idx = ops.cast(
            ops.argmax(predictions[:, 0:num_bin], axis=-1), "int32"
        )
        bin_idx_float = ops.cast(bin_idx, dtype=predictions.dtype)
        residual_norm = ops.take_along_axis(
            predictions[:, num_bin : num_bin * 2],
            ops.expand_dims(bin_idx, axis=-1),
            axis=-1,
        )[:, 0]

        # Divide 2pi into equal sized bins to compute the angle per class/bin.
        angle_per_class = (2 * np.pi) / num_bin
        residual_angle = residual_norm * (angle_per_class / 2)

        # bin_center is computed using the bin_idx and angle_per class,
        # (e.g., 0, 30, 60, 90, 120, ..., 270, 300, 330). Then residual is
        # added.
        heading = ops.mod(
            bin_idx_float * angle_per_class + residual_angle, 2 * np.pi
        )
        heading_mask = heading > np.pi
        heading = ops.where(heading_mask, heading - 2 * np.pi, heading)
        return heading


def decode_bin_box(pd, num_head_bin, anchor_size):
    """Decode bin based box encoding."""
    with keras.backend.name_scope("decode_bin_box"):
        delta = []
        start = 0
        for dim in [0, 1, 2]:
            delta.append(pd[:, start])
            start = start + 1

        heading = decode_bin_heading(pd[:, start:], num_head_bin)
        start = start + num_head_bin * 2

        size_res_norm = pd[:, start : start + 3]
        # [N,3]
        lwh = ops.cast(
            size_res_norm
            * ops.array(list(anchor_size), dtype=size_res_norm.dtype)
            + ops.array(list(anchor_size), dtype=size_res_norm.dtype),
            pd.dtype,
        )

        loc = ops.stack(delta, axis=-1)
        box = ops.concatenate(
            [loc, lwh, ops.expand_dims(heading, axis=-1)], axis=-1
        )
        return box


@keras_cv_export("keras_cv.layers.HeatmapDecoder")
class HeatmapDecoder(keras.layers.Layer):
    """A Keras layer that decodes predictions of a 3d object detection model.

    Arg:
      class_id: the integer index for a particular class.
      num_head_bin: number of bin classes divided by [-2pi, 2pi].
      anchor_size: the size of anchor at each xyz dimension.
      max_pool_size: the 2d pooling size for heatmap.
      max_num_box: top number of boxes select from heatmap.
      heatmap_threshold: the threshold to set a heatmap as positive.
      voxel_size: the x, y, z dimension of each voxel.
      spatial_size: the x, y, z boundary of voxels.
    """

    def __init__(
        self,
        class_id,
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
        self.class_id = class_id
        self.num_head_bin = num_head_bin
        self.anchor_size = anchor_size
        self.max_pool_size = max_pool_size
        self.max_num_box = max_num_box
        self.heatmap_threshold = heatmap_threshold
        self.voxel_size = voxel_size
        self.spatial_size = spatial_size
        self.built = True

    def call(self, prediction):
        """Accepts raw predictions, and returns decoded boxes.

        Args:
            prediction: float Tensor.
        """
        heatmap = ops.softmax(prediction[..., :2])[..., 1:2]
        heatmap_pool = ops.max_pool(heatmap, self.max_pool_size, 1, "same")
        heatmap_mask = heatmap > self.heatmap_threshold
        heatmap_local_maxima_mask = ops.equal(heatmap, heatmap_pool)
        # [B, H, W, 1]
        heatmap_mask = ops.logical_and(heatmap_mask, heatmap_local_maxima_mask)
        # [B, H, W, 1]
        heatmap = ops.where(heatmap_mask, heatmap, 0)
        # [B, H, W]
        heatmap = ops.squeeze(heatmap, axis=-1)

        b, h, w = ops.shape(heatmap)
        heatmap = ops.reshape(heatmap, [b, h * w])
        _, top_index = ops.top_k(heatmap, k=self.max_num_box)

        # [B, H, W, ?]
        box_prediction = prediction[:, :, :, 2:]
        f = box_prediction.shape[-1]
        box_prediction = ops.reshape(box_prediction, [b, h * w, f])
        heatmap = ops.reshape(heatmap, [b, h * w])
        # [B, max_num_box, ?]
        box_prediction = ops.take_along_axis(
            box_prediction, ops.expand_dims(top_index, axis=-1), axis=1
        )
        # [B, max_num_box]
        box_score = ops.take_along_axis(heatmap, top_index, axis=1)
        box_class = ops.ones_like(box_score, "int32") * self.class_id
        # [B*max_num_box, ?]
        f = ops.shape(box_prediction)[-1]
        box_prediction_reshape = ops.reshape(
            box_prediction, [b * self.max_num_box, f]
        )
        # [B*max_num_box, 7]
        box_decoded = decode_bin_box(
            box_prediction_reshape, self.num_head_bin, self.anchor_size
        )
        # [B, max_num_box, 7]
        box_decoded = ops.reshape(box_decoded, [b, self.max_num_box, 7])
        global_xyz = ops.zeros([b, 3])
        ref_xyz = voxel_utils.compute_feature_map_ref_xyz(
            self.voxel_size, self.spatial_size, global_xyz
        )
        # [B, H, W, 3]
        ref_xyz = ops.squeeze(ref_xyz, axis=-2)
        f = list(ref_xyz.shape)[-1]
        ref_xyz = ops.reshape(ref_xyz, [b, h * w, f])
        # [B, max_num_box, 3]
        ref_xyz = ops.take_along_axis(
            ref_xyz, ops.expand_dims(top_index, axis=-1), axis=1
        )

        box_decoded_cxyz = ops.cast(
            ref_xyz + box_decoded[:, :, :3], box_decoded.dtype
        )
        box_decoded = ops.concatenate(
            [box_decoded_cxyz, box_decoded[:, :, 3:]], axis=-1
        )
        return box_decoded, box_class, box_score

    def get_config(self):
        config = {
            "class_id": self.class_id,
            "num_head_bin": self.num_head_bin,
            "anchor_size": self.anchor_size,
            "max_pool_size": self.max_pool_size,
            "max_num_box": self.max_num_box,
            "heatmap_threshold": self.heatmap_threshold,
            "voxel_size": self.voxel_size,
            "spatial_size": self.spatial_size,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
