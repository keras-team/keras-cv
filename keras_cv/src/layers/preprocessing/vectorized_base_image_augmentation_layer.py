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
import tree

from keras_cv.src import bounding_box
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import config
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.backend import scope
from keras_cv.src.utils import preprocessing

H_AXIS = -3
W_AXIS = -2

IMAGES = "images"
LABELS = "labels"
TARGETS = "targets"
BOUNDING_BOXES = "bounding_boxes"
KEYPOINTS = "keypoints"
SEGMENTATION_MASKS = "segmentation_masks"

IS_DICT = "is_dict"
BATCHED = "batched"
USE_TARGETS = "use_targets"


@keras_cv_export("keras_cv.layers.VectorizedBaseImageAugmentationLayer")
class VectorizedBaseImageAugmentationLayer(keras.layers.Layer):
    """Abstract base layer for vectorized image augmentation.

    This layer contains base functionalities for preprocessing layers which
    augment image related data, e.g. image and in the future, label and bounding
    boxes. The subclasses could avoid making certain mistakes and reduce code
    duplications.

    This layer requires you to implement one method: `augment_images()`, which
    augments one single image during the training. There are a few additional
    methods that you can implement for added functionality on the layer:

    `augment_labels()`, which handles label augmentation if the layer supports
    that.

    `augment_bounding_boxes()`, which handles the bounding box augmentation, if
    the layer supports that.

    `get_random_transformations()`, which should produce a batch of random
    transformation settings. The transformation object, which must be a batched
    Tensor or a dictionary where each input is a batched Tensor, will be passed
    to `augment_images`, `augment_labels` and `augment_bounding_boxes`, to
    coordinate the randomness behavior, eg, in the RandomFlip layer, the image
    and bounding_boxes should be changed in the same way.

    The `call()` method support two formats of inputs:
    1. Single image tensor with 3D (HWC) or 4D (NHWC) format.
    2. A dict of tensors with stable keys. The supported keys are:
      `"images"`, `"labels"` and `"bounding_boxes"` at the moment. We might add
      more keys in future when we support more types of augmentation.

    The output of the `call()` will be in two formats, which will be the same
    structure as the inputs.

    The `call()` will unpack the inputs, forward to the correct function, and
    pack the output back to the same structure as the inputs.

    By default, the dense or ragged status of the output will be preserved.
    However, you can override this behavior by setting
    `self.force_output_dense_images = True`,
    `self.force_output_dense_segmentation_masks = True` in your `__init__()`
    method. When enabled, images and segmentation masks will be converted to
    dense tensor by `to_tensor()` if ragged.

    ```python
    class SubclassLayer(VectorizedBaseImageAugmentationLayer):
      def __init__(self):
        super().__init__()
        self.force_output_dense_images = True
        self.force_output_dense_segmentation_masks = True
    ```

    Note that since the randomness is also a common functionality, this layer
    also includes a keras_backend.RandomGenerator, which can be used to
    produce the random numbers. The random number generator is stored in the
    `self._random_generator` attribute.
    """

    def __init__(self, seed=None, **kwargs):
        super().__init__(**kwargs)
        if seed:
            self._random_generator = tf.random.Generator.from_seed(seed=seed)
        else:
            self._random_generator = tf.random.get_global_generator()
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True

    @property
    def force_output_dense_images(self):
        """Control whether to force outputting of dense images."""
        return getattr(self, "_force_output_dense_images", False)

    @force_output_dense_images.setter
    def force_output_dense_images(self, force_output_dense_images):
        self._force_output_dense_images = force_output_dense_images

    @property
    def force_output_dense_segmentation_masks(self):
        """Control whether to force outputting of dense segmentation masks."""
        return getattr(self, "_force_output_dense_segmentation_masks", False)

    @force_output_dense_segmentation_masks.setter
    def force_output_dense_segmentation_masks(
        self, force_output_dense_segmentation_masks
    ):
        self._force_output_dense_segmentation_masks = (
            force_output_dense_segmentation_masks
        )

    def augment_ragged_image(self, image, transformation, **kwargs):
        """Augment an image from a ragged image batch during training.

        This method accepts a single Dense image Tensor, and returns a Dense
        image. The resulting images are then stacked back into a ragged image
        batch. The behavior of this method should be identical to that of
        `augment_images()` but is to operate on a batch-wise basis.

        Args:
            image: a single image from the batch
            transformation: a single transformation sampled from
                `get_random_transformations()`.
            kwargs: all the other call arguments (i.e. bounding_boxes, labels,
                etc.).
        Returns:
            Augmented image.
        """
        raise NotImplementedError(
            "A ragged image batch was passed to layer of type "
            f"`{type(self).__name__}`. This layer does not implement "
            "`augment_ragged_image()`. If this is a `keras_cv`, open a GitHub "
            "issue requesting Ragged functionality on the layer titled: "
            f"'`{type(self).__name__}`: ragged image support'. If this is a "
            "custom layer, implement the `augment_ragged_image()` method."
        )

    def compute_ragged_image_signature(self, images):
        """Computes the output image signature for the `augment_image()`
        function.

        Must be overridden to return tensors with different shapes than the
        input images. By default, returns either a `tf.RaggedTensorSpec`
        matching the input image spec, or a `tf.TensorSpec` matching the input
        image spec.
        """
        ragged_spec = tf.RaggedTensorSpec(
            shape=images.shape[1:],
            ragged_rank=1,
            dtype=self.compute_dtype,
        )
        return ragged_spec

    def augment_images(self, images, transformations, **kwargs):
        """Augment a batch of images during training.

        Args:
          images: 4D image input tensor to the layer. Forwarded from
            `layer.call()`. This should generally have the shape [B, H, W, C].
            Forwarded from `layer.call()`.
          transformations: The transformations object produced by
            `get_random_transformations`. Used to coordinate the randomness
            between image, label, bounding box, keypoints, and segmentation
            mask.

        Returns:
          output 4D tensor, which will be forward to `layer.call()`.
        """
        raise NotImplementedError()

    def augment_labels(self, labels, transformations, **kwargs):
        """Augment a batch of  labels during training.

        Args:
          labels: 2D label to the layer. Forwarded from `layer.call()`.
          transformations: The transformations object produced by
            `get_random_transformations`. Used to coordinate the randomness
            between image, label, bounding box, keypoints, and segmentation
            mask.

        Returns:
          output 2D tensor, which will be forward to `layer.call()`.
        """
        raise NotImplementedError()

    def augment_targets(self, targets, transformations, **kwargs):
        """Augment a batch of targets during training.

        Args:
          targets: 2D label to the layer. Forwarded from `layer.call()`.
          transformations: The transformations object produced by
            `get_random_transformations`. Used to coordinate the randomness
            between image, label, bounding box, keypoints, and segmentation
            mask.

        Returns:
          output 2D tensor, which will be forward to `layer.call()`.
        """
        return self.augment_labels(targets, transformations, **kwargs)

    def augment_bounding_boxes(self, bounding_boxes, transformations, **kwargs):
        """Augment bounding boxes for one image during training.

        Args:
          bounding_boxes: 3D bounding boxes to the layer. Forwarded from
            `call()`.
          transformations: The transformations object produced by
            `get_random_transformations`. Used to coordinate the randomness
            between image, label, bounding box, keypoints, and segmentation
            mask.

        Returns:
          output 3D tensor, which will be forward to `layer.call()`.
        """
        raise NotImplementedError()

    def augment_keypoints(self, keypoints, transformations, **kwargs):
        """Augment a batch of keypoints for one image during training.

        Args:
          keypoints: 3D keypoints input tensor to the layer. Forwarded from
            `layer.call()`. Shape should be [batch, num_keypoints, 2] in the
            specified keypoint format.
          transformations: The transformations object produced by
            `get_random_transformations`. Used to coordinate the randomness
            between image, label, bounding box, keypoints, and segmentation
            mask.

        Returns:
          output 3D tensor, which will be forward to `layer.call()`.
        """
        raise NotImplementedError()

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        """Augment a batch of images' segmentation masks during training.

        Args:
          segmentation_masks: 4D segmentation mask input tensor to the layer.
            This should generally have the shape [B, H, W, 1], or in some cases
            [B, H, W, C] for multilabeled data. Forwarded from `layer.call()`.
          transformations: The transformations object produced by
            `get_random_transformations`. Used to coordinate the randomness
            between image, label, bounding box, keypoints, and segmentation
            mask.

        Returns:
          output 4D tensor containing the augmented segmentation mask, which
          will be forward to `layer.call()`.
        """
        raise NotImplementedError()

    def get_random_transformation_batch(
        self,
        batch_size,
        images=None,
        labels=None,
        bounding_boxes=None,
        keypoints=None,
        segmentation_masks=None,
    ):
        """Produce random transformations config for a batch of inputs.

        This is used to produce same randomness between
        image/label/bounding_box.

        Args:
          batch_size: the batch size of transformations configuration to sample.
          images: 3D image tensor from inputs.
          labels: optional 1D label tensor from inputs.
          bounding_boxes: optional 2D bounding boxes tensor from inputs.
          segmentation_masks: optional 3D segmentation mask tensor from inputs.

        Returns:
          Any type of object, which will be forwarded to `augment_images`,
          `augment_labels` and `augment_bounding_boxes` as the `transformations`
          parameter.
        """
        # Required to work with map_fn in the ragged cast.
        return tf.zeros((batch_size))

    def _unwrap_ragged_image_call(self, inputs):
        images = inputs.get(IMAGES, None)
        labels = inputs.get(LABELS, None)
        bounding_boxes = inputs.get(BOUNDING_BOXES, None)
        keypoints = inputs.get(KEYPOINTS, None)
        segmentation_masks = inputs.get(SEGMENTATION_MASKS, None)
        transformation = inputs.get("transformations")
        images = images.to_tensor()
        images = self.augment_ragged_image(
            image=images,
            label=labels,
            bounding_boxes=bounding_boxes,
            keypoints=keypoints,
            segmentation_mask=segmentation_masks,
            transformation=transformation,
        )
        return tf.RaggedTensor.from_tensor(images)

    def _batch_augment(self, inputs):
        images = inputs.get(IMAGES, None)
        raw_images = images
        labels = inputs.get(LABELS, None)
        bounding_boxes = inputs.get(BOUNDING_BOXES, None)
        keypoints = inputs.get(KEYPOINTS, None)
        segmentation_masks = inputs.get(SEGMENTATION_MASKS, None)

        batch_size = tf.shape(images)[0]

        transformations = self.get_random_transformation_batch(
            batch_size,
            images=images,
            labels=labels,
            bounding_boxes=bounding_boxes,
            keypoints=keypoints,
            segmentation_masks=segmentation_masks,
        )

        if isinstance(images, tf.RaggedTensor):
            inputs_for_raggeds = {"transformations": transformations, **inputs}
            images = tf.map_fn(
                self._unwrap_ragged_image_call,
                inputs_for_raggeds,
                fn_output_signature=self.compute_ragged_image_signature(images),
            )
        else:
            images = self.augment_images(
                images,
                transformations=transformations,
                bounding_boxes=bounding_boxes,
                labels=labels,
            )
        if (
            isinstance(images, tf.RaggedTensor)
            and self.force_output_dense_images
        ):
            images = images.to_tensor()
        result = {IMAGES: images}

        if labels is not None:
            labels = self.augment_targets(
                labels,
                transformations=transformations,
                bounding_boxes=bounding_boxes,
                images=images,
                raw_images=raw_images,
            )
            result[LABELS] = labels

        if bounding_boxes is not None:
            bounding_boxes = self.augment_bounding_boxes(
                bounding_boxes,
                transformations=transformations,
                labels=labels,
                images=images,
                raw_images=raw_images,
            )
            bounding_boxes = bounding_box.to_ragged(bounding_boxes)
            result[BOUNDING_BOXES] = bounding_boxes

        if keypoints is not None:
            keypoints = self.augment_keypoints(
                keypoints,
                transformations=transformations,
                labels=labels,
                bounding_boxes=bounding_boxes,
                images=images,
                raw_images=raw_images,
            )
            result[KEYPOINTS] = keypoints

        if segmentation_masks is not None:
            segmentation_masks = self.augment_segmentation_masks(
                segmentation_masks,
                transformations=transformations,
                labels=labels,
                bounding_boxes=bounding_boxes,
                images=images,
                raw_images=raw_images,
            )
            if (
                isinstance(segmentation_masks, tf.RaggedTensor)
                and self.force_output_dense_segmentation_masks
            ):
                segmentation_masks = segmentation_masks.to_tensor()
            result[SEGMENTATION_MASKS] = segmentation_masks

        # preserve any additional inputs unmodified by this layer.
        for key in inputs.keys() - result.keys():
            result[key] = inputs[key]
        return result

    def call(self, inputs):
        # try to convert a given backend native tensor to TensorFlow tensor
        # before passing it over to TFDataScope
        is_tf_backend = config.backend() == "tensorflow"
        is_in_tf_graph = not tf.executing_eagerly()
        contains_ragged = lambda y: any(
            tree.map_structure(
                lambda x: isinstance(x, (tf.RaggedTensor, tf.SparseTensor)),
                tree.flatten(y),
            )
        )
        inputs_contain_ragged = contains_ragged(inputs)
        if not is_tf_backend and not inputs_contain_ragged:
            inputs = tree.map_structure(
                lambda x: tf.convert_to_tensor(x), inputs
            )
        with scope.TFDataScope():
            inputs = self._ensure_inputs_are_compute_dtype(inputs)
            inputs, metadata = self._format_inputs(inputs)
            images = inputs[IMAGES]
            if images.shape.rank == 3 or images.shape.rank == 4:
                outputs = self._format_output(
                    self._batch_augment(inputs), metadata
                )
            else:
                raise ValueError(
                    "Image augmentation layers are expecting inputs to be "
                    "rank 3 (HWC) or 4D (NHWC) tensors. Got shape: "
                    f"{images.shape}"
                )
        # convert the outputs to backend native tensors if none of them
        # contain RaggedTensors. Note that if the user passed in Raggeds
        # but the outputs are dense, we still don't want to convert to
        # backend native tensors. This is to avoid breaking TF data
        # pipelines that can't easily be ported to become backend
        # agnostic.
        if not is_tf_backend and not is_in_tf_graph:
            if not inputs_contain_ragged and not contains_ragged(outputs):
                outputs = tree.map_structure(
                    # some layers return None, handle that case when
                    # converting to tensors
                    lambda x: ops.convert_to_tensor(x) if x is not None else x,
                    outputs,
                )
        return outputs

    def _format_inputs(self, inputs):
        metadata = {IS_DICT: True, USE_TARGETS: False}
        if tf.is_tensor(inputs):
            # single image input tensor
            metadata[IS_DICT] = False
            inputs = {IMAGES: inputs}
        else:
            # Copy the input dict before we mutate it.
            inputs = dict(inputs)

        metadata[BATCHED] = inputs["images"].shape.rank == 4
        if inputs["images"].shape.rank == 3:
            for key in list(inputs.keys()):
                if key == BOUNDING_BOXES:
                    inputs[BOUNDING_BOXES]["boxes"] = tf.expand_dims(
                        inputs[BOUNDING_BOXES]["boxes"], axis=0
                    )
                    inputs[BOUNDING_BOXES]["classes"] = tf.expand_dims(
                        inputs[BOUNDING_BOXES]["classes"], axis=0
                    )
                else:
                    inputs[key] = tf.expand_dims(inputs[key], axis=0)

        if not isinstance(inputs, dict):
            raise ValueError(
                "Expect the inputs to be image tensor or dict. Got "
                f"inputs={inputs}"
            )

        if BOUNDING_BOXES in inputs:
            inputs[BOUNDING_BOXES] = self._format_bounding_boxes(
                inputs[BOUNDING_BOXES]
            )

        if isinstance(inputs, dict) and TARGETS in inputs:
            # TODO(scottzhu): Check if it only contains the valid keys
            inputs[LABELS] = inputs[TARGETS]
            del inputs[TARGETS]
            metadata[USE_TARGETS] = True
            return inputs, metadata

        return inputs, metadata

    def _format_output(self, output, metadata):
        if not metadata[BATCHED]:
            for key in list(output.keys()):
                if key == BOUNDING_BOXES:
                    output[BOUNDING_BOXES]["boxes"] = tf.squeeze(
                        output[BOUNDING_BOXES]["boxes"], axis=0
                    )
                    output[BOUNDING_BOXES]["classes"] = tf.squeeze(
                        output[BOUNDING_BOXES]["classes"], axis=0
                    )
                else:
                    output[key] = tf.squeeze(output[key], axis=0)

        if not metadata[IS_DICT]:
            return output[IMAGES]
        elif metadata[USE_TARGETS]:
            output[TARGETS] = output[LABELS]
            del output[LABELS]
        return output

    def _ensure_inputs_are_compute_dtype(self, inputs):
        if not isinstance(inputs, dict):
            return preprocessing.ensure_tensor(
                inputs,
                self.compute_dtype,
            )
        # Copy the input dict before we mutate it.
        inputs = dict(inputs)
        inputs[IMAGES] = preprocessing.ensure_tensor(
            inputs[IMAGES],
            self.compute_dtype,
        )
        if LABELS in inputs:
            inputs[LABELS] = preprocessing.ensure_tensor(
                inputs[LABELS],
                self.compute_dtype,
            )
        if KEYPOINTS in inputs:
            inputs[KEYPOINTS] = preprocessing.ensure_tensor(
                inputs[KEYPOINTS],
                self.compute_dtype,
            )
        if SEGMENTATION_MASKS in inputs:
            inputs[SEGMENTATION_MASKS] = preprocessing.ensure_tensor(
                inputs[SEGMENTATION_MASKS],
                self.compute_dtype,
            )
        if BOUNDING_BOXES in inputs:
            inputs[BOUNDING_BOXES]["boxes"] = preprocessing.ensure_tensor(
                inputs[BOUNDING_BOXES]["boxes"],
                self.compute_dtype,
            )
            inputs[BOUNDING_BOXES]["classes"] = preprocessing.ensure_tensor(
                inputs[BOUNDING_BOXES]["classes"],
                self.compute_dtype,
            )
        return inputs

    def _format_bounding_boxes(self, bounding_boxes):
        # We can't catch the case where this is None, sometimes RaggedTensor
        # drops this dimension.
        if "classes" not in bounding_boxes:
            raise ValueError(
                "Bounding boxes are missing class_id. If you would like to pad "
                "the bounding boxes with class_id, use: "
                "`bounding_boxes['classes'] = "
                "tf.ones_like(bounding_boxes['boxes'])`."
            )
        return bounding_boxes
