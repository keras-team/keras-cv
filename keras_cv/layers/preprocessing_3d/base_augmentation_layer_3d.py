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

POINT_CLOUDS = "point_clouds"
BOUNDING_BOXES = "bounding_boxes"
OBJECT_POINT_CLOUDS = "object_point_clouds"
OBJECT_BOUNDING_BOXES = "object_bounding_boxes"
ADDITIONAL_POINT_CLOUDS = "additional_point_clouds"
ADDITIONAL_BOUNDING_BOXES = "additional_bounding_boxes"
BOX_LABEL_INDEX = 7
POINTCLOUD_LABEL_INDEX = 3
POINTCLOUD_FEATURE_INDEX = 4


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class BaseAugmentationLayer3D(tf.keras.__internal__.layers.BaseRandomLayer):
    """Abstract base layer for data augmentaion for 3D preception.

    This layer contains base functionalities for preprocessing layers which
    augment 3D preception related data, eg. point_clouds and in future, images.
    The subclasses could avoid making certain mistakes and reduce code
    duplications.

    This layer requires you to implement one method: `augment_point_clouds()`,
    which augments one or a sequence of point clouds during the training. There are a few
    additional methods that you can implement for added functionality on the
    layer:

    `augment_bounding_boxes()`, which handles the bounding box augmentation, if
    the layer supports that.

    `get_random_transformation()`, which should produce a random transformation
    setting. The tranformation object, which could be any type, will be passed
    to `augment_point_clouds` and `augment_bounding_boxes`, to
    coodinate the randomness behavior, eg, in the RotateZ layer, the point_clouds
    and bounding_boxes should be changed in the same way.

    The `call()` method support two formats of inputs:
    1. A dict of tensors with stable keys. The supported keys are:
      `"point_clouds"` and `"bounding_boxes"` at the moment. We might add
      more keys in future when we support more types of augmentation.

    The output of the `call()` will be in two formats, which will be the same
    structure as the inputs.

    The `call()` will handle the logic detecting the training/inference mode,
    unpack the inputs, forward to the correct function, and pack the output back
    to the same structure as the inputs.

    By default the `call()` method leverages the `tf.vectorized_map()` function.
    Auto-vectorization can be disabled by setting `self.auto_vectorize = False`
    in your `__init__()` method.  When disabled, `call()` instead relies
    on `tf.map_fn()`. For example:

    ```python
    class SubclassLayer(keras_cv.BaseImageAugmentationLayer):
      def __init__(self):
        super().__init__()
        self.auto_vectorize = False
    ```

    Example:

    ```python
    class RandomRotateZ(keras_cv.BaseImageAugmentationLayer):

      def __init__(self, max_rotation, **kwargs):
        super().__init__(**kwargs)
        self._max_rotation = max_rotation

      def augment_pointclouds(self, point_clouds, transformation):
        pose = transformation['pos']
        # Rotate points.
        pointcloud_xyz = geometry.CoordinateTransform(pointcloud[..., :3], pose)
        pointcloud = tf.concat([pointcloud_xyz, pointcloud[..., 3:]], axis=-1)
        return pointcloud, boxes
    ```

    Note that since the randomness is also a common functionnality, this layer
    also includes a tf.keras.backend.RandomGenerator, which can be used to
    produce the random numbers.  The random number generator is stored in the
    `self._random_generator` attribute.
    """

    def __init__(self, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.auto_vectorize = False

    @property
    def auto_vectorize(self):
        """Control whether automatic vectorization occurs.

        By default the `call()` method leverages the `tf.vectorized_map()`
        function.  Auto-vectorization can be disabled by setting
        `self.auto_vectorize = False` in your `__init__()` method.  When
        disabled, `call()` instead relies on `tf.map_fn()`. For example:

        ```python
        class SubclassLayer(BaseImageAugmentationLayer):
          def __init__(self):
            super().__init__()
            self.auto_vectorize = False
        ```
        """
        return getattr(self, "_auto_vectorize", True)

    @auto_vectorize.setter
    def auto_vectorize(self, auto_vectorize):
        self._auto_vectorize = auto_vectorize

    @property
    def _map_fn(self):
        if self.auto_vectorize:
            return tf.vectorized_map
        else:
            return tf.map_fn

    def augment_point_clouds_bounding_boxes(
        self, point_clouds, bounding_boxes, transformation, **kwargs
    ):
        """Augment a single point cloud frame during training.

        Args:
          point_clouds: 3D point cloud input tensor to the layer. Forwarded from
            `layer.call()`.
          bounding_boxes: 3D bounding boxes to the layer. Forwarded from
            `call()`.
          transformation: The transformation object produced by
            `get_random_transformation`. Used to coordinate the randomness
            between point clouds, bounding boxs.

        Returns:
          output 3D tensor, which will be forward to `layer.call()`.
        """
        raise NotImplementedError()

    def get_random_transformation(self, point_clouds=None, bounding_boxes=None):
        """Produce random transformation config for one single input.

        This is used to produce same randomness between
        image/label/bounding_box.

        Args:
          point_clouds: 3D point clouds tensor from inputs.
          bounding_box: 3D bounding boxes tensor from inputs.

        Returns:
          Any type of object, which will be forwarded to `augment_point_clouds`,
          and `augment_bounding_box` as the `transformation` parameter.
        """
        return None

    def call(self, inputs, training=True):
        if training:
            point_clouds = inputs[POINT_CLOUDS]
            bounding_boxes = inputs[BOUNDING_BOXES]
            if point_clouds.shape.rank == 3 and bounding_boxes.shape.rank == 3:
                return self._augment(inputs)
            elif point_clouds.shape.rank == 4 and bounding_boxes.shape.rank == 4:
                return self._batch_augment(inputs)
            else:
                raise ValueError(
                    "Point clouds augmentation layers are expecting inputs point clouds and bounding boxes to "
                    "be rank 3D (Frame, Point, Feature) or 4D (Batch, Frame, Point, Feature) tensors. Got shape: {} and {}".format(
                        point_clouds.shape, bounding_boxes.shape
                    )
                )
        else:
            return inputs

    def _augment(self, inputs):
        point_clouds = inputs.get(POINT_CLOUDS, None)
        bounding_boxes = inputs.get(BOUNDING_BOXES, None)
        transformation = self.get_random_transformation(
            point_clouds=point_clouds,
            bounding_boxes=bounding_boxes,
        )
        point_clouds, bounding_boxes = self.augment_point_clouds_bounding_boxes(
            point_clouds,
            bounding_boxes=bounding_boxes,
            transformation=transformation,
        )

        result = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}

        # preserve any additional inputs unmodified by this layer.
        for key in inputs.keys() - result.keys():
            result[key] = inputs[key]
        return result

    def _batch_augment(self, inputs):
        return self._map_fn(self._augment, inputs)
