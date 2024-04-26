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

from keras_cv.src import bounding_box
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)


@keras_cv_export("keras_cv.layers.MixUp")
class MixUp(BaseImageAugmentationLayer):
    """MixUp implements the MixUp data augmentation technique.

    Args:
        alpha: Float between 0 and 1. Inverse scale parameter for the gamma
            distribution. This controls the shape of the distribution from which
            the smoothing values are sampled. Defaults to 0.2, which is a
            recommended value when training an imagenet1k classification model.
        seed: Integer. Used to create a random seed.

    References:
        - [MixUp paper](https://arxiv.org/abs/1710.09412).
        - [MixUp for Object Detection paper](https://arxiv.org/pdf/1902.04103).

    Example:
    ```python
    (images, labels), _ = keras.datasets.cifar10.load_data()
    images, labels = images[:10], labels[:10]
    # Labels must be floating-point and one-hot encoded
    labels = tf.cast(tf.one_hot(labels, 10), tf.float32)
    mixup = keras_cv.layers.preprocessing.MixUp(10)
    augmented_images, updated_labels = mixup(
        {'images': images, 'labels': labels}
    )
    # output == {'images': updated_images, 'labels': updated_labels}
    ```
    """

    def __init__(self, alpha=0.2, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.alpha = alpha
        self.seed = seed

    def _sample_from_beta(self, alpha, beta, shape):
        sample_alpha = tf.random.gamma(
            shape,
            alpha=alpha,
        )
        sample_beta = tf.random.gamma(
            shape,
            alpha=beta,
        )
        return sample_alpha / (sample_alpha + sample_beta)

    def _batch_augment(self, inputs):
        self._validate_inputs(inputs)
        images = inputs.get("images", None)
        labels = inputs.get("labels", None)
        bounding_boxes = inputs.get("bounding_boxes", None)
        segmentation_masks = inputs.get("segmentation_masks", None)
        images, lambda_sample, permutation_order = self._mixup(images)
        if labels is not None:
            labels = self._update_labels(
                tf.cast(labels, dtype=self.compute_dtype),
                lambda_sample,
                permutation_order,
            )
            inputs["labels"] = labels
        if bounding_boxes is not None:
            bounding_boxes = self._update_bounding_boxes(
                bounding_boxes, permutation_order
            )
            inputs["bounding_boxes"] = bounding_boxes
        inputs["images"] = images
        if segmentation_masks is not None:
            segmentation_masks = self._update_segmentation_masks(
                segmentation_masks, lambda_sample, permutation_order
            )
            inputs["segmentation_masks"] = segmentation_masks
        return inputs

    def _augment(self, inputs):
        raise ValueError(
            "MixUp received a single image to `call`. The layer relies on "
            "combining multiple examples, and as such will not behave as "
            "expected. Please call the layer with 2 or more samples."
        )

    def _mixup(self, images):
        batch_size = tf.shape(images)[0]
        permutation_order = tf.random.shuffle(
            tf.range(0, batch_size), seed=self.seed
        )

        lambda_sample = self._sample_from_beta(
            self.alpha, self.alpha, (batch_size,)
        )
        lambda_sample = tf.cast(
            tf.reshape(lambda_sample, [-1, 1, 1, 1]), dtype=self.compute_dtype
        )

        mixup_images = tf.cast(
            tf.gather(images, permutation_order), dtype=self.compute_dtype
        )

        images = lambda_sample * images + (1.0 - lambda_sample) * mixup_images

        return images, tf.squeeze(lambda_sample), permutation_order

    def _update_labels(self, labels, lambda_sample, permutation_order):
        labels_for_mixup = tf.gather(labels, permutation_order)

        lambda_sample = tf.reshape(lambda_sample, [-1, 1])

        labels = (
            lambda_sample * labels + (1.0 - lambda_sample) * labels_for_mixup
        )

        return labels

    def _update_bounding_boxes(self, bounding_boxes, permutation_order):
        boxes, classes = bounding_boxes["boxes"], bounding_boxes["classes"]
        boxes_for_mixup = tf.gather(boxes, permutation_order)
        classes_for_mixup = tf.gather(classes, permutation_order)
        boxes = tf.concat([boxes, boxes_for_mixup], axis=1)
        classes = tf.concat([classes, classes_for_mixup], axis=1)
        return {"boxes": boxes, "classes": classes}

    def _update_segmentation_masks(
        self, segmentation_masks, lambda_sample, permutation_order
    ):
        lambda_sample = tf.reshape(lambda_sample, [-1, 1, 1, 1])

        segmentation_masks_for_mixup = tf.gather(
            segmentation_masks, permutation_order
        )

        segmentation_masks = (
            lambda_sample * segmentation_masks
            + (1.0 - lambda_sample) * segmentation_masks_for_mixup
        )

        return segmentation_masks

    def _validate_inputs(self, inputs):
        images = inputs.get("images", None)
        labels = inputs.get("labels", None)
        bounding_boxes = inputs.get("bounding_boxes", None)
        segmentation_masks = inputs.get("segmentation_masks", None)

        if images is None or (
            labels is None
            and bounding_boxes is None
            and segmentation_masks is None
        ):
            raise ValueError(
                "MixUp expects inputs in a dictionary with format "
                '{"images": images, "labels": labels}. or'
                '{"images": images, "bounding_boxes": bounding_boxes}. or'
                '{"images": images, "segmentation_masks": segmentation_masks}. '
                f"Got: inputs = {inputs}."
            )

        if labels is not None and not labels.dtype.is_floating:
            raise ValueError(
                f"MixUp received labels with type {labels.dtype}. "
                "Labels must be of type float."
            )

        if bounding_boxes is not None:
            _ = bounding_box.validate_format(bounding_boxes)

        if segmentation_masks is not None:
            if len(segmentation_masks.shape) != 4:
                raise ValueError(
                    "MixUp expects shape of segmentation_masks as "
                    "[batch, h, w, num_classes]. "
                    f"Got: shape = {segmentation_masks.shape}. "
                )

    def get_config(self):
        config = {
            "alpha": self.alpha,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
