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


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class FourierMix(BaseImageAugmentationLayer):
    """FourierMix implements the FMix data augmentation technique.

    Args:
        alpha: Float value for beta distribution.  Inverse scale parameter for the gamma
            distribution.  This controls the shape of the distribution from which the
            smoothing values are sampled.  Defaults to 0.5, which is a recommended value
            in the paper.
        decay_power: A float value representing the decay power.  Defaults to 3, as
            recommended in the paper.
        seed: Integer. Used to create a random seed.
    References:
        [FMix paper](https://arxiv.org/abs/2002.12047).

    Sample usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    fourier_mix = keras_cv.layers.preprocessing.FourierMix(0.5)
    augmented_images, updated_labels = fourier_mix({'images': images, 'labels': labels})
    # output == {'images': updated_images, 'labels': updated_labels}
    ```
    """

    def __init__(self, alpha=0.5, decay_power=3, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.alpha = alpha
        self.decay_power = decay_power
        self.seed = seed

    def _sample_from_beta(self, alpha, beta, shape):
        sample_alpha = tf.random.gamma(
            shape, 1.0, beta=alpha, seed=self._random_generator.make_legacy_seed()
        )
        sample_beta = tf.random.gamma(
            shape, 1.0, beta=beta, seed=self._random_generator.make_legacy_seed()
        )
        return sample_alpha / (sample_alpha + sample_beta)

    @staticmethod
    def _fftfreq(signal_size, sample_spacing=1):
        """This function returns the sample frequencies of a discrete fourier transform.
        The result array contains the frequency bin centers starting at 0 using the
        sample spacing.
        """

        results = tf.concat(
            [
                tf.range((signal_size - 1) / 2 + 1, dtype=tf.int32),
                tf.range(-(signal_size // 2), 0, dtype=tf.int32),
            ],
            0,
        )

        return results / (signal_size * sample_spacing)

    def _apply_fftfreq(self, h, w):
        # Applying the fourier transform across 2 dimensions (height and width).
        fx = FourierMix._fftfreq(w)[: w // 2 + 1 + w % 2]
        fy = FourierMix._fftfreq(h)
        fy = tf.expand_dims(fy, -1)

        return tf.math.sqrt(fx * fx + fy * fy)

    def _get_spectrum(self, freqs, decay_power, channel, h, w):
        # Function to apply a low pass filter by decaying its high frequency components.
        scale = tf.ones(1) / tf.cast(
            tf.math.maximum(freqs, tf.convert_to_tensor([1 / tf.reduce_max([w, h])]))
            ** decay_power,
            tf.float32,
        )

        param_size = tf.concat(
            [tf.constant([channel]), tf.shape(freqs), tf.constant([2])], 0
        )
        param = self._random_generator.random_normal(param_size)

        scale = tf.expand_dims(scale, -1)[None, :]

        return scale * param

    def _sample_mask_from_transform(self, decay, shape, ch=1):
        # Sampling low frequency map from fourier transform.
        freqs = self._apply_fftfreq(shape[0], shape[1])
        spectrum = self._get_spectrum(freqs, decay, ch, shape[0], shape[1])
        spectrum = tf.complex(spectrum[:, 0], spectrum[:, 1])

        mask = tf.math.real(tf.signal.irfft2d(spectrum, shape))
        mask = mask[:1, : shape[0], : shape[1]]

        mask = mask - tf.reduce_min(mask)
        mask = mask / tf.reduce_max(mask)
        return mask

    def _binarise_mask(self, mask, lam, in_shape):
        # Create the final mask from the sampled values.
        idx = tf.argsort(tf.reshape(mask, [-1]), direction="DESCENDING")
        mask = tf.reshape(mask, [-1])
        num = tf.cast(tf.math.round(lam * tf.cast(tf.size(mask), tf.float32)), tf.int32)

        updates = tf.concat(
            [
                tf.ones((num,), tf.float32),
                tf.zeros((tf.size(mask) - num,), tf.float32),
            ],
            0,
        )

        mask = tf.scatter_nd(
            tf.expand_dims(idx, -1), updates, tf.expand_dims(tf.size(mask), -1)
        )

        mask = tf.reshape(mask, in_shape)
        return mask

    def _batch_augment(self, inputs):
        images = inputs.get("images", None)
        labels = inputs.get("labels", None)
        if images is None or labels is None:
            raise ValueError(
                "FourierMix expects inputs in a dictionary with format "
                '{"images": images, "labels": labels}.'
                f"Got: inputs = {inputs}"
            )
        images, lambda_sample, permutation_order = self._fourier_mix(images)
        if labels is not None:
            labels = self._update_labels(labels, lambda_sample, permutation_order)
            inputs["labels"] = labels
        inputs["images"] = images
        return inputs

    def _augment(self, inputs):
        raise ValueError(
            "FourierMix received a single image to `call`.  The layer relies on "
            "combining multiple examples, and as such will not behave as "
            "expected.  Please call the layer with 2 or more samples."
        )

    def _fourier_mix(self, images):
        shape = tf.shape(images)
        permutation_order = tf.random.shuffle(tf.range(0, shape[0]), seed=self.seed)

        lambda_sample = self._sample_from_beta(self.alpha, self.alpha, (shape[0],))

        # generate masks utilizing mapped calls
        masks = tf.map_fn(
            lambda x: self._sample_mask_from_transform(self.decay_power, shape[1:-1]),
            tf.range(shape[0], dtype=tf.float32),
        )

        # binarise masks utilizing mapped calls
        masks = tf.map_fn(
            lambda i: self._binarise_mask(masks[i], lambda_sample[i], shape[1:-1]),
            tf.range(shape[0], dtype=tf.int32),
            fn_output_signature=tf.float32,
        )
        masks = tf.expand_dims(masks, -1)

        fmix_images = tf.gather(images, permutation_order)
        images = masks * images + (1.0 - masks) * fmix_images

        return images, lambda_sample, permutation_order

    def _update_labels(self, labels, lambda_sample, permutation_order):
        labels_for_fmix = tf.gather(labels, permutation_order)

        # for broadcasting
        batch_size = tf.expand_dims(tf.shape(labels)[0], -1)
        labels_rank = tf.rank(labels)
        broadcast_shape = tf.concat([batch_size, tf.ones(labels_rank - 1, tf.int32)], 0)
        lambda_sample = tf.reshape(lambda_sample, broadcast_shape)

        labels = lambda_sample * labels + (1.0 - lambda_sample) * labels_for_fmix
        return labels

    def get_config(self):
        config = {
            "alpha": self.alpha,
            "decay_power": self.decay_power,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
