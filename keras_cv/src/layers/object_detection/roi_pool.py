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


from keras_cv.src import bounding_box
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops


@keras_cv_export("keras_cv.layers.ROIPooler")
class ROIPooler(keras.layers.Layer):
    """
    Pooling feature map of dynamic shape into region of interest (ROI) of fixed
    shape.

    Mainly used in Region CNN (RCNN) networks. This works for a single-level
    input feature map.

    This layer splits the feature map into [target_size[0], target_size[1]]
    areas, and performs max pooling for each area. The area coordinates will be
    quantized.

    Args:
        bounding_box_format: a case-insensitive string.
            For detailed information on the supported format, see the
            [KerasCV bounding box documentation](https://keras.io/api/keras_cv/bounding_box/formats/).
        target_size: List or Tuple of 2 integers of the pooled shape
        image_shape: List of Tuple of 3 integers, or `TensorShape` of the input
            image shape.

    Example:
    ```python
    feature_map = tf.random.normal([2, 16, 16, 512])
    roi_pooler = ROIPooler(bounding_box_format="yxyx", target_size=[7, 7],
      image_shape=[224, 224, 3])
    rois = tf.constant([[[15., 30., 25., 45.]], [[22., 1., 30., 32.]]])
    pooled_feature_map = roi_pooler(feature_map, rois)
    ```
    """  # noqa: E501

    def __init__(
        self,
        bounding_box_format,
        # TODO(consolidate size vs shape for KPL and here)
        target_size,
        image_shape,
        **kwargs,
    ):
        if not isinstance(target_size, (tuple, list)):
            raise ValueError(
                "Expected `target_size` to be tuple or list, got "
                f"{type(target_size)}"
            )
        if len(target_size) != 2:
            raise ValueError(
                f"Expected `target_size` to be size 2, got {len(target_size)}"
            )
        if (
            image_shape[0] is None
            or image_shape[1] is None
            or image_shape[2] is None
        ):
            raise ValueError(
                f"`image_shape` cannot have dynamic shape, got {image_shape}"
            )
        super().__init__(**kwargs)
        self.bounding_box_format = bounding_box_format
        self.target_height = target_size[0]
        self.target_width = target_size[1]
        self.image_shape = image_shape
        self.built = True

    def call(self, feature_map, rois):
        """
        Args:
          feature_map: [batch_size, H, W, C] float Tensor, the feature map
            extracted from image.
          rois: [batch_size, N, 4] float Tensor, the region of interests to be
            pooled.
        Returns:
          pooled_feature_map: [batch_size, N, target_size, C] float Tensor
        """
        # convert to relative format given feature map shape != image shape
        rois = bounding_box.convert_format(
            rois,
            source=self.bounding_box_format,
            target="rel_yxyx",
            image_shape=self.image_shape,
        )
        pooled_feature_map = list()
        for batch_idx in range(ops.shape(feature_map)[0]):
            pooled_feature_map.append(
                self._pool_single_sample(
                    args=[feature_map[batch_idx], rois[batch_idx]]
                )
            )
        return ops.stack(pooled_feature_map, axis=0)

    def _pool_single_sample(self, args):
        """
        Args: tuple of
          feature_map: [H, W, C] float Tensor
          rois: [N, 4] float Tensor
        Returns:
          pooled_feature_map: [N, target_height, target_width, C] float Tensor
        """
        feature_map, rois = args
        num_rois = ops.shape(rois)[0]
        height, width, channel = ops.shape(feature_map)
        regions = []
        # TODO (consider vectorize it for better performance)
        for n in range(num_rois):
            # [4]
            roi = rois[n, :]
            y_start = height * roi[0]
            x_start = width * roi[1]
            region_height = height * (roi[2] - roi[0])
            region_width = width * (roi[3] - roi[1])
            h_step = region_height / self.target_height
            w_step = region_width / self.target_width
            region_steps = []
            for i in range(self.target_height):
                for j in range(self.target_width):
                    height_start = y_start + i * h_step
                    height_end = height_start + h_step
                    height_start = ops.cast(height_start, "int32")
                    height_end = ops.cast(height_end, "int32")
                    # if feature_map shape smaller than roi, h_step would be 0
                    # in this case the result will be feature_map[0, 0, ...]
                    height_end = height_start + ops.maximum(
                        1, height_end - height_start
                    )
                    width_start = x_start + j * w_step
                    width_end = width_start + w_step
                    width_start = ops.cast(width_start, "int32")
                    width_end = ops.cast(width_end, "int32")
                    width_end = width_start + ops.maximum(
                        1, width_end - width_start
                    )
                    # [h_step, w_step, C]
                    region_step = ops.slice(
                        inputs=feature_map,
                        start_indices=[
                            height_start,
                            width_start,
                            ops.convert_to_tensor(0, dtype="int32"),
                        ],
                        shape=[
                            height_end - height_start,
                            width_end - width_start,
                            ops.shape(feature_map)[-1],
                        ],
                    )
                    region_steps.append(ops.max(region_step, axis=[0, 1]))
            regions.append(
                ops.reshape(
                    ops.stack(region_steps),
                    [self.target_height, self.target_width, channel],
                )
            )
        return ops.stack(regions)

    def get_config(self):
        config = {
            "bounding_box_format": self.bounding_box_format,
            "target_size": [self.target_height, self.target_width],
            "image_shape": self.image_shape,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
