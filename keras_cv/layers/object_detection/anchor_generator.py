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

import tensorflow as tf
from tensorflow import keras

from keras_cv import bounding_box


class AnchorGenerator(keras.layers.Layer):
    """AnchorGenerator generates anchors for multiple feature maps.

    AnchorGenerator takes multiple scales and generates anchor boxes based on the anchor
    sizes, scales, aspect ratios, and strides provided.  To invoke AnchorGenerator, call
    it on the image that needs anchor boxes.

    `sizes` and `strides` must match structurally - they are pairs.  Scales and
    aspect ratios can either be a list, that is then used for all of the sizes
    (aka levels), or a dictionary from `{'level_{number}': [parameters at scale...]}`.

    Args:
      bounding_box_format: The format of bounding boxes to generate. Refer
        [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
        for more details on supported bounding box formats.
      sizes: A list of integers that represent the anchor sizes for each level,
        or a dictionary of integer lists with each key representing a level.
        For each anchor size, anchor height will be `anchor_size / sqrt(aspect_ratio)`,
        and anchor width will be `anchor_size * sqrt(aspect_ratio)`.  This is repeated
        for each scale and aspect ratio.
      scales: A list of floats corresponding to multipliers that will be
        multiplied by each `anchor_size` to generate a level.
      aspect_ratios: A list of floats representing the ratio of anchor width to height.
      strides: iterable of ints that represent the anchor stride size between
        center of anchors at each scale.
      clip_boxes: Whether or not to clip generated anchor boxes to the image size.
        Defaults to `False`.

    Usage:
    ```python
    strides = [8, 16, 32]
    scales = [1, 1.2599210498948732, 1.5874010519681994]
    sizes = [32.0, 64.0, 128.0]
    aspect_ratios = [0.5, 1.0, 2.0]

    image = tf.random.uniform((512, 512, 3))
    anchor_generator = cv_layers.AnchorGenerator(
        bounding_box_format="rel_yxyx",
        sizes=sizes,
        aspect_ratios=aspect_ratios,
        scales=scales,
        strides=strides,
        clip_boxes=True,
    )
    anchors = anchor_generator(image)
    print(anchors)
    # > {0: ..., 1: ..., 2: ...}
    ```

    Input shape: an image with shape `[H, W, C]`
    Output: a dictionary with integer keys corresponding to each level of the feature
        pyramid.  The size of the anchors at each level will be
        `(H/strides[i] * W/strides[i] * len(scales) * len(aspect_ratios), 4)`.
    """

    def __init__(
        self,
        bounding_box_format,
        sizes,
        scales,
        aspect_ratios,
        strides,
        clip_boxes=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bounding_box_format = bounding_box_format
        # aspect_ratio is a single list that is the same across all levels.
        sizes, strides = self._format_sizes_and_strides(sizes, strides)
        aspect_ratios = self._match_param_structure_to_sizes(aspect_ratios, sizes)
        scales = self._match_param_structure_to_sizes(scales, sizes)

        self.anchor_generators = {}
        for k in sizes.keys():
            self.anchor_generators[k] = _SingleAnchorGenerator(
                bounding_box_format,
                sizes[k],
                scales[k],
                aspect_ratios[k],
                strides[k],
                clip_boxes,
                dtype=self.compute_dtype,
            )
        self.built = True

    @staticmethod
    def _format_sizes_and_strides(sizes, strides):
        result_sizes = AnchorGenerator._ensure_param_is_levels_dict(sizes, "sizes")
        result_strides = AnchorGenerator._ensure_param_is_levels_dict(
            strides, "strides"
        )

        if sorted(result_strides.keys()) != sorted(result_sizes.keys()):
            raise ValueError(
                "Expected sizes and strides to be either lists of"
                "the same length, or dictionaries with the same keys.  Received "
                f"sizes={sizes}, strides={strides}"
            )

        return result_sizes, result_strides

    @staticmethod
    def _ensure_param_is_levels_dict(param, param_name):
        """Takes a param and its name, converts lists to dictionaries of levels.
        For example, the list [1, 2] is converted to {0: 1, 1: 2}.

        Raises:
            ValueError: when param is not a dict, list or tuple.
        """
        if isinstance(param, dict):
            return param
        if not isinstance(param, (list, tuple)):
            raise ValueError(
                f"Expected {param_name} to be a dict, list or tuple, received "
                f"{param_name}={param}"
            )

        result = {}
        for i in range(len(param)):
            result[i] = param[i]
        return result

    @staticmethod
    def _match_param_structure_to_sizes(params, sizes):
        """broadcast the params to match sizes."""
        # if isinstance(sizes, (tuple, list)):
        #     return [params] * len(sizes)
        if not isinstance(sizes, dict):
            raise ValueError(
                "the structure of `sizes` must be a dict, " f"received sizes={sizes}"
            )

        return tf.nest.map_structure(lambda _: params, sizes)

    def __call__(self, image=None, image_shape=None):
        if image is None and image_shape is None:
            raise ValueError("AnchorGenerator() requires `images` or `image_shape`.")

        if image is not None:
            if image.shape.rank != 3:
                raise ValueError(
                    "Expected `image` to be a Tensor of rank 3.  Got "
                    f"image.shape.rank={image.shape.rank}"
                )
            image_shape = tf.shape(image)

        anchor_generators = tf.nest.flatten(self.anchor_generators)
        results = [anchor_gen(image_shape) for anchor_gen in anchor_generators]
        results = tf.nest.pack_sequence_as(self.anchor_generators, results)
        for key in results:
            results[key] = bounding_box.convert_format(
                results[key],
                source="yxyx",
                target=self.bounding_box_format,
                image_shape=image_shape,
            )
        return results


# TODO(tanzheny): consider having customized anchor offset.
class _SingleAnchorGenerator:
    """Internal utility to generate anchors for a single feature map in `yxyx` format.

    Example:
    ```python
    anchor_gen = _SingleAnchorGenerator(32, [.5, 1., 2.], stride=16)
    anchors = anchor_gen([512, 512, 3])
    ```

    Input shape: the size of the image, `[H, W, C]`
    Output shape: the size of anchors, `[(H / stride) * (W / stride), 4]`

    Args:
      sizes: A single int represents the base anchor size. The anchor
        height will be `anchor_size / sqrt(aspect_ratio)`, anchor width will be
        `anchor_size * sqrt(aspect_ratio)`.
      scales: A list/tuple, or a list/tuple of a list/tuple of positive
        floats representing the actual anchor size to the base `anchor_size`.
      aspect_ratios: a list/tuple of positive floats representing the ratio of
        anchor width to anchor height.
      stride: A single int represents the anchor stride size between center of
        each anchor.
      clip_boxes: Boolean to represent whether the anchor coordinates should be
        clipped to the image size. Defaults to `False`.
      dtype: (Optional) The data type to use for the output anchors.  Defaults to
        'float32'.

    """

    def __init__(
        self,
        bounding_box_format,
        sizes,
        scales,
        aspect_ratios,
        stride,
        clip_boxes=False,
        dtype="float32",
    ):

        self.sizes = sizes
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.stride = stride
        self.clip_boxes = clip_boxes
        self.dtype = dtype

    def __call__(self, image_size):
        image_height = tf.cast(image_size[0], tf.float32)
        image_width = tf.cast(image_size[1], tf.float32)

        aspect_ratios = tf.cast(self.aspect_ratios, tf.float32)
        aspect_ratios_sqrt = tf.cast(tf.sqrt(aspect_ratios), dtype=tf.float32)
        anchor_size = tf.cast(self.sizes, tf.float32)

        # [K]
        anchor_heights = []
        anchor_widths = []
        for scale in self.scales:
            anchor_size_t = anchor_size * scale
            anchor_height = anchor_size_t / aspect_ratios_sqrt
            anchor_width = anchor_size_t * aspect_ratios_sqrt
            anchor_heights.append(anchor_height)
            anchor_widths.append(anchor_width)
        anchor_heights = tf.concat(anchor_heights, axis=0)
        anchor_widths = tf.concat(anchor_widths, axis=0)
        half_anchor_heights = tf.reshape(0.5 * anchor_heights, [1, 1, -1])
        half_anchor_widths = tf.reshape(0.5 * anchor_widths, [1, 1, -1])

        stride = tf.cast(self.stride, tf.float32)
        # [W]
        cx = tf.range(0.5 * stride, image_width + 1, stride)
        # [H]
        cy = tf.range(0.5 * stride, image_height + 1, stride)
        # [H, W]
        cx_grid, cy_grid = tf.meshgrid(cx, cy)
        # [H, W, 1]
        cx_grid = tf.expand_dims(cx_grid, axis=-1)
        cy_grid = tf.expand_dims(cy_grid, axis=-1)

        y_min = tf.reshape(cy_grid - half_anchor_heights, (-1,))
        y_max = tf.reshape(cy_grid + half_anchor_heights, (-1,))
        x_min = tf.reshape(cx_grid - half_anchor_widths, (-1,))
        x_max = tf.reshape(cx_grid + half_anchor_widths, (-1,))

        # [H * W * K, 1]
        y_min = tf.expand_dims(y_min, axis=-1)
        y_max = tf.expand_dims(y_max, axis=-1)
        x_min = tf.expand_dims(x_min, axis=-1)
        x_max = tf.expand_dims(x_max, axis=-1)

        if self.clip_boxes:
            y_min = tf.maximum(tf.minimum(y_min, image_height), 0.0)
            y_max = tf.maximum(tf.minimum(y_max, image_height), 0.0)
            x_min = tf.maximum(tf.minimum(x_min, image_width), 0.0)
            x_max = tf.maximum(tf.minimum(x_max, image_width), 0.0)

        # [H * W * K, 4]
        return tf.cast(tf.concat([y_min, x_min, y_max, x_max], axis=-1), self.dtype)
