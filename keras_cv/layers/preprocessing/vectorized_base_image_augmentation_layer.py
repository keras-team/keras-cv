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
SEGMENTATION_MASKS = "segmentation_masks"

IS_DICT = "is_dict"
BATCHED = "batched"
USE_TARGETS = "use_targets"

_ALL_KEYS = [IMAGES, LABELS, TARGETS, BOUNDING_BOXES, KEYPOINTS, SEGMENTATION_MASKS]


class VectorizedBaseImageAugmentationLayer(
    tf.keras.__internal__.layers.BaseRandomLayer
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)

    def augment_images(self, images, transformations, **kwargs):
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

    def augment_labels(self, labels, transformation, **kwargs):
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

    def augment_targets(self, targets, transformations, **kwargs):
        """Augment a single target during training.

        Args:
          target: 1D label to the layer. Forwarded from `layer.call()`.
          transformation: The transformation object produced by
            `get_random_transformation`. Used to coordinate the randomness
            between image, label, bounding box, keypoints, and segmentation mask.

        Returns:
          output 1D tensor, which will be forward to `layer.call()`.
        """
        return self.augment_labels(targets, transformations)

    def augment_bounding_boxes(self, bounding_boxes, transformations, **kwargs):
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

    def augment_keypoints(self, keypoints, transformations, **kwargs):
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

    def augment_segmentation_masks(self, segmentation_masks, transformations, **kwargs):
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

    def get_random_transformation_batch(
        self,
        batch_size,
        image=None,
        label=None,
        bounding_boxes=None,
        keypoints=None,
        segmentation_mask=None,
    ):
        """Produce random transformation config for a batch of inputs.

        This is used to produce same randomness between
        image/label/bounding_box.

        Args:
          batch_size: the batch size of transformation configuration to sample.
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

    def _batch_augment(self, inputs):
        images = inputs.get(IMAGES, None)
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

        images = self.augment_images(
            images,
            transformations=transformations,
            bounding_boxes=bounding_boxes,
            label=labels,
        )

        result = {IMAGES: images}
        if labels is not None:
            labels = self.augment_targets(
                labels,
                transformations=transformations,
                bounding_boxes=bounding_boxes,
                image=images,
            )
            result[LABELS] = labels

        if bounding_boxes is not None:
            bounding_boxes = self.augment_bounding_boxes(
                bounding_boxes,
                transformations=transformations,
                labels=labels,
                images=raw_images,
            )
            bounding_boxes = bounding_box.to_ragged(bounding_boxes)
            result[BOUNDING_BOXES] = bounding_boxes

        if keypoints is not None:
            keypoints = self.augment_keypoints(
                keypoints,
                transformation=transformations,
                label=labels,
                bounding_boxes=bounding_boxes,
                images=images,
            )
            result[KEYPOINTS] = keypoints
        if segmentation_masks is not None:
            segmentation_masks = self.augment_segmentation_masks(
                segmentation_masks,
                transformations=transformations,
            )
            result[SEGMENTATION_MASKS] = segmentation_masks

        # preserve any additional inputs unmodified by this layer.
        for key in inputs.keys() - result.keys():
            result[key] = inputs[key]
        return result

    def call(self, inputs, training=True):
        inputs = self._ensure_inputs_are_compute_dtype(inputs)
        if training:
            inputs, metadata = self._format_inputs(inputs)
            images = inputs[IMAGES]
            if images.shape.rank == 3 or images.shape.rank == 4:
                return self._format_output(self._batch_augment(inputs), metadata)
            else:
                raise ValueError(
                    "Image augmentation layers are expecting inputs to be "
                    "rank 3 (HWC) or 4D (NHWC) tensors. Got shape: "
                    f"{images.shape}"
                )
        else:
            return inputs

    def _format_inputs(self, inputs):
        metadata = {IS_DICT: True, USE_TARGETS: False}
        if tf.is_tensor(inputs):
            # single image input tensor
            metadata[IS_DICT] = False
            inputs = {IMAGES: inputs}

        metadata[BATCHED] = inputs["images"].shape.rank == 4
        if inputs["images"].shape.rank == 3:
            for key in list(inputs.keys()):
                inputs[key] = tf.expand_dims(inputs[key], axis=0)

        if not isinstance(inputs, dict):
            raise ValueError(
                f"Expect the inputs to be image tensor or dict. Got inputs={inputs}"
            )

        if BOUNDING_BOXES in inputs:
            inputs[BOUNDING_BOXES] = self._format_bounding_boxes(inputs[BOUNDING_BOXES])

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
        inputs[IMAGES] = preprocessing.ensure_tensor(
            inputs[IMAGES],
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
