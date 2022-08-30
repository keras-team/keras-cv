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

from keras_cv import bounding_box
from keras_cv.utils import preprocessing

# In order to support both unbatched and batched inputs, the horizontal
# and verticle axis is reverse indexed
H_AXIS = -3
W_AXIS = -2

IMAGES = "images"
LABELS = "labels"
TARGETS = "targets"
BOUNDING_BOXES = "bounding_boxes"
KEYPOINTS = "keypoints"
RAGGED_BOUNDING_BOXES = "ragged_bounding_boxes"
SEGMENTATION_MASK = "segmentation_mask"
IS_DICT = "is_dict"
USE_TARGETS = "use_targets"


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class BaseImageAugmentationLayer(tf.keras.__internal__.layers.BaseRandomLayer):
    """Abstract base layer for image augmentaion.

    This layer contains base functionalities for preprocessing layers which
    augment image related data, eg. image and in future, label and bounding
    boxes.  The subclasses could avoid making certain mistakes and reduce code
    duplications.

    This layer requires you to implement one method: `augment_image()`, which
    augments one single image during the training. There are a few additional
    methods that you can implement for added functionality on the layer:

    `augment_label()`, which handles label augmentation if the layer supports
    that.

    `augment_bounding_boxes()`, which handles the bounding box augmentation, if
    the layer supports that.

    `get_random_transformation()`, which should produce a random transformation
    setting. The tranformation object, which could be any type, will be passed
    to `augment_image`, `augment_label` and `augment_bounding_boxes`, to
    coodinate the randomness behavior, eg, in the RandomFlip layer, the image
    and bounding_boxes should be changed in the same way.

    The `call()` method support two formats of inputs:
    1. Single image tensor with 3D (HWC) or 4D (NHWC) format.
    2. A dict of tensors with stable keys. The supported keys are:
      `"images"`, `"labels"` and `"bounding_boxes"` at the moment. We might add
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
    class RandomContrast(keras_cv.BaseImageAugmentationLayer):

      def __init__(self, factor=(0.5, 1.5), **kwargs):
        super().__init__(**kwargs)
        self._factor = factor

      def augment_image(self, image, transformation):
        random_factor = tf.random.uniform([], self._factor[0], self._factor[1])
        mean = tf.math.reduced_mean(inputs, axis=-1, keep_dim=True)
        return (inputs - mean) * random_factor + mean
    ```

    Note that since the randomness is also a common functionnality, this layer
    also includes a tf.keras.backend.RandomGenerator, which can be used to
    produce the random numbers.  The random number generator is stored in the
    `self._random_generator` attribute.
    """

    def __init__(self, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)

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

    def augment_image(self, image, transformation, **kwargs):
        """Augment a single image during training.

        Args:
          image: 3D image input tensor to the layer. Forwarded from
            `layer.call()`.
          transformation: The transformation object produced by
            `get_random_transformation`. Used to coordinate the randomness
            between image, label, bounding box, keypoints, and segmentation mask.

        Returns:
          output 3D tensor, which will be forward to `layer.call()`.
        """
        raise NotImplementedError()

    def augment_label(self, label, transformation, **kwargs):
        """Augment a single label during training.

        Args:
          label: 1D label to the layer. Forwarded from `layer.call()`.
          transformation: The transformation object produced by
            `get_random_transformation`. Used to coordinate the randomness
            between image, label, bounding box, keypoints, and segmentation mask.

        Returns:
          output 1D tensor, which will be forward to `layer.call()`.
        """
        raise NotImplementedError()

    def augment_target(self, target, transformation, **kwargs):
        """Augment a single target during training.

        Args:
          target: 1D label to the layer. Forwarded from `layer.call()`.
          transformation: The transformation object produced by
            `get_random_transformation`. Used to coordinate the randomness
            between image, label, bounding box, keypoints, and segmentation mask.

        Returns:
          output 1D tensor, which will be forward to `layer.call()`.
        """
        return self.augment_label(target, transformation)

    def augment_bounding_boxes(self, bounding_boxes, transformation, **kwargs):
        """Augment bounding boxes for one image during training.

        Args:
          image: 3D image input tensor to the layer. Forwarded from
            `layer.call()`.
          bounding_boxes: 2D bounding boxes to the layer. Forwarded from
            `call()`.
          transformation: The transformation object produced by
            `get_random_transformation`. Used to coordinate the randomness
            between image, label, bounding box, keypoints, and segmentation mask.

        Returns:
          output 2D tensor, which will be forward to `layer.call()`.
        """
        raise NotImplementedError()

    def augment_keypoints(self, keypoints, transformation, **kwargs):
        """Augment keypoints for one image during training.

        Args:
          keypoints: 2D keypoints input tensor to the layer. Forwarded from
            `layer.call()`.
          transformation: The transformation object produced by
            `get_random_transformation`. Used to coordinate the randomness
            between image, label, bounding box, keypoints, and segmentation mask.

        Returns:
          output 2D tensor, which will be forward to `layer.call()`.
        """
        raise NotImplementedError()

    def augment_segmentation_mask(self, segmentation_mask, transformation, **kwargs):
        """Augment a single image's segmentation mask during training.

        Args:
          segmentation_mask: 3D segmentation mask input tensor to the layer.
            This should generally have the shape [H, W, 1], or in some cases [H, W, C] for multilabeled data.
            Forwarded from `layer.call()`.
          transformation: The transformation object produced by
            `get_random_transformation`. Used to coordinate the randomness
            between image, label, bounding box, keypoints, and segmentation mask.

        Returns:
          output 3D tensor containing the augmented segmentation mask, which will be forward to `layer.call()`.
        """
        raise NotImplementedError()

    def get_random_transformation(
        self,
        image=None,
        label=None,
        bounding_boxes=None,
        keypoints=None,
        segmentation_mask=None,
    ):
        """Produce random transformation config for one single input.

        This is used to produce same randomness between
        image/label/bounding_box.

        Args:
          image: 3D image tensor from inputs.
          label: optional 1D label tensor from inputs.
          bounding_box: optional 2D bounding boxes tensor from inputs.
          segmentation_mask: optional 3D segmentation mask tensor from inputs.

        Returns:
          Any type of object, which will be forwarded to `augment_image`,
          `augment_label` and `augment_bounding_box` as the `transformation`
          parameter.
        """
        return None

    def call(self, inputs, training=True):
        inputs = self._ensure_inputs_are_compute_dtype(inputs)
        if training:
            inputs, metadata = self._format_inputs(inputs)
            images = inputs[IMAGES]
            if images.shape.rank == 3:
                return self._format_output(self._augment(inputs), metadata)
            elif images.shape.rank == 4:
                return self._format_output(self._batch_augment(inputs), metadata)
            else:
                raise ValueError(
                    "Image augmentation layers are expecting inputs to be "
                    "rank 3 (HWC) or 4D (NHWC) tensors. Got shape: "
                    f"{images.shape}"
                )
        else:
            return inputs

    def _augment(self, inputs):
        image = inputs.get(IMAGES, None)
        label = inputs.get(LABELS, None)
        bounding_boxes = inputs.get(BOUNDING_BOXES, None)
        keypoints = inputs.get(KEYPOINTS, None)
        segmentation_mask = inputs.get(SEGMENTATION_MASK, None)
        transformation = self.get_random_transformation(
            image=image,
            label=label,
            bounding_boxes=bounding_boxes,
            keypoints=keypoints,
            segmentation_mask=segmentation_mask,
        )
        image = self.augment_image(
            image,
            transformation=transformation,
            bounding_boxes=bounding_boxes,
            label=label,
        )
        result = {IMAGES: image}
        if label is not None:
            label = self.augment_target(
                label,
                transformation=transformation,
                bounding_boxes=bounding_boxes,
                image=image,
            )
            result[LABELS] = label
        if bounding_boxes is not None:
            bounding_boxes = self.augment_bounding_boxes(
                bounding_boxes,
                transformation=transformation,
                label=label,
                image=image,
            )
            result[BOUNDING_BOXES] = bounding_boxes
        if keypoints is not None:
            keypoints = self.augment_keypoints(
                keypoints,
                transformation=transformation,
                label=label,
                bounding_boxes=bounding_boxes,
                image=image,
            )
            result[KEYPOINTS] = keypoints
        if segmentation_mask is not None:
            segmentation_mask = self.augment_segmentation_mask(
                segmentation_mask,
                transformation=transformation,
            )
            result[SEGMENTATION_MASK] = segmentation_mask

        # preserve any additional inputs unmodified by this layer.
        for key in inputs.keys() - result.keys():
            result[key] = inputs[key]
        return result

    def _batch_augment(self, inputs):
        return self._map_fn(self._augment, inputs)

    def _format_inputs(self, inputs):
        metadata = {IS_DICT: True, USE_TARGETS: False}
        if tf.is_tensor(inputs):
            # single image input tensor
            metadata[IS_DICT] = False
            inputs = {IMAGES: inputs}
            return inputs, metadata

        if not isinstance(inputs, dict):
            raise ValueError(
                f"Expect the inputs to be image tensor or dict. Got inputs={inputs}"
            )

        if BOUNDING_BOXES in inputs:
            inputs[BOUNDING_BOXES], updates = self._format_bounding_boxes(
                inputs[BOUNDING_BOXES]
            )
            metadata.update(updates)

        if isinstance(inputs, dict) and TARGETS in inputs:
            # TODO(scottzhu): Check if it only contains the valid keys
            inputs[LABELS] = inputs[TARGETS]
            del inputs[TARGETS]
            metadata[USE_TARGETS] = True
            return inputs, metadata

        return inputs, metadata

    def _format_bounding_boxes(self, bounding_boxes):
        metadata = {RAGGED_BOUNDING_BOXES: False}
        if isinstance(bounding_boxes, tf.RaggedTensor):
            metadata = {RAGGED_BOUNDING_BOXES: True}
            bounding_boxes = bounding_box.pad_with_sentinels(bounding_boxes)

        if bounding_boxes.shape[-1] < 5:
            raise ValueError(
                "Bounding boxes are missing class_id. If you would like to pad the "
                "bounding boxes with class_id, use `keras_cv.bounding_box.add_class_id`"
            )
        return bounding_boxes, metadata

    def _format_output(self, output, metadata):
        if not metadata[IS_DICT]:
            return output[IMAGES]
        elif metadata[USE_TARGETS]:
            output[TARGETS] = output[LABELS]
            del output[LABELS]

        if BOUNDING_BOXES in output:
            if metadata[RAGGED_BOUNDING_BOXES]:
                output[BOUNDING_BOXES] = bounding_box.filter_sentinels(
                    output[BOUNDING_BOXES]
                )
        return output

    def _ensure_inputs_are_compute_dtype(self, inputs):
        if isinstance(inputs, dict):
            inputs[IMAGES] = preprocessing.ensure_tensor(
                inputs[IMAGES],
                self.compute_dtype,
            )
        else:
            inputs = preprocessing.ensure_tensor(
                inputs,
                self.compute_dtype,
            )
        return inputs
