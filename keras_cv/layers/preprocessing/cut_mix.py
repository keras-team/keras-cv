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

from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.utils import fill_utils


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class CutMix(BaseImageAugmentationLayer):
    """CutMix implements the CutMix data augmentation technique.

    Args:
        alpha: Float between 0 and 1.  Inverse scale parameter for the gamma
            distribution.  This controls the shape of the distribution from which the
            smoothing values are sampled.  Defaults 1.0, which is a recommended value
            when training an imagenet1k classification model.
        seed: Integer. Used to create a random seed.
    References:
       [CutMix paper]( https://arxiv.org/abs/1905.04899).

    Sample usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    labels = tf.one_hot(labels.squeeze(), 10)

    cutmix = keras_cv.layers.preprocessing.cut_mix.CutMix(10)
    output = cutmix({"images": images[:32], "labels": labels[:32]})
    # output == {'images': updated_images, 'labels': updated_labels}
    ```
    """

    def __init__(self, alpha=1.0, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.alpha = alpha
        self.seed = seed

    def _sample_from_beta(self, alpha, beta, shape):
        sample_alpha = tf.random.gamma(
            shape, 1.0, beta=alpha, seed=self._random_generator.make_legacy_seed()
        )
        sample_beta = tf.random.gamma(
            shape, 1.0, beta=beta, seed=self._random_generator.make_legacy_seed()
        )
        return sample_alpha / (sample_alpha + sample_beta)

    def _batch_augment(self, inputs):
        self._validate_inputs(inputs)
        images = inputs.get("images", None)
        labels = inputs.get("labels", None)
        if images is None or labels is None:
            raise ValueError(
                "CutMix expects inputs in a dictionary with format "
                '{"images": images, "labels": labels}.'
                f"Got: inputs = {inputs}"
            )
        images, labels = self._update_labels(*self._cutmix(images, labels))
        inputs["images"] = images
        inputs["labels"] = labels
        return inputs

    def _augment(self, inputs):
        raise ValueError(
            "CutMix received a single image to `call`.  The layer relies on "
            "combining multiple examples, and as such will not behave as "
            "expected.  Please call the layer with 2 or more samples."
        )

    def _cutmix(self, images, labels):
        """Apply cutmix."""
        input_shape = tf.shape(images)
        batch_size, image_height, image_width = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )

        permutation_order = tf.random.shuffle(tf.range(0, batch_size), seed=self.seed)
        lambda_sample = self._sample_from_beta(self.alpha, self.alpha, (batch_size,))

        ratio = tf.math.sqrt(1 - lambda_sample)

        cut_height = tf.cast(
            ratio * tf.cast(image_height, dtype=tf.float32), dtype=tf.int32
        )
        cut_width = tf.cast(
            ratio * tf.cast(image_height, dtype=tf.float32), dtype=tf.int32
        )

        random_center_height = tf.random.uniform(
            shape=[batch_size], minval=0, maxval=image_height, dtype=tf.int32
        )
        random_center_width = tf.random.uniform(
            shape=[batch_size], minval=0, maxval=image_width, dtype=tf.int32
        )

        bounding_box_area = cut_height * cut_width
        lambda_sample = 1.0 - bounding_box_area / (image_height * image_width)
        lambda_sample = tf.cast(lambda_sample, dtype=tf.float32)

        images = fill_utils.fill_rectangle(
            images,
            random_center_width,
            random_center_height,
            cut_width,
            cut_height,
            tf.gather(images, permutation_order),
        )

        return images, labels, lambda_sample, permutation_order

    def _update_labels(self, images, labels, lambda_sample, permutation_order):
        cutout_labels = tf.gather(labels, permutation_order)

        lambda_sample = tf.reshape(lambda_sample, [-1, 1])
        labels = lambda_sample * labels + (1.0 - lambda_sample) * cutout_labels
        return images, labels

    def _validate_inputs(self, inputs):
        labels = inputs.get("labels", None)
        if labels is None:
            raise ValueError(
                "CutMix expects 'labels' to be present in its inputs. "
                "CutMix relies on both images an labels. "
                "Please pass a dictionary with keys 'images' "
                "containing the image Tensor, and 'labels' containing "
                "the classification labels. "
                "For example, `cut_mix({'images': images, 'labels': labels})`."
            )
        if not labels.dtype.is_floating:
            raise ValueError(
                f"CutMix received labels with type {labels.dtype}. "
                "Labels must be of type float."
            )

    def get_config(self):
        config = {
            "alpha": self.alpha,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
