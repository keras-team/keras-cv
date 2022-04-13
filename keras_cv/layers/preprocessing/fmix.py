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


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class FMix(tf.keras.__internal__.layers.BaseImageAugmentationLayer):
    """FMix implements the FMix data augmentation technique.

    Args:
        alpha: Float value for beta distribution.  Inverse scale parameter for the gamma
            distribution.  This controls the shape of the distribution from which the
            smoothing values are sampled.  Defaults 0.5, which is a recommended value
            in the paper.
        decay_power: A float value representing the decay power for frequency decay prop
            1/f**d.  Defaults to 3, as recommended in the paper.
        max_soft: Float value representing softening value between 0 and 0.5 which
            smooths hard edges in the mask.  Defaults to 0.0.
        seed: Integer. Used to create a random seed.
    References:
        [FMix paper](https://arxiv.org/abs/2002.12047).

    Sample usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    fmix = keras_cv.layers.preprocessing.mix_up.FMix(0.5)
    augmented_images, updated_labels = fmix({'images': images, 'labels': labels})
    # output == {'images': updated_images, 'labels': updated_labels}
    ```
    """

    def __init__(self, alpha=0.2, decay_power=3, max_soft=0.0, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.alpha = alpha
        self.decay_power = decay_power
        self.max_soft = max_soft
        self.seed = seed

    @staticmethod
    def _sample_from_beta(alpha, beta):
        sample_alpha = tf.random.gamma((), 1.0, beta=alpha)
        sample_beta = tf.random.gamma((), 1.0, beta=beta)
        return sample_alpha / (sample_alpha + sample_beta)

    def fftfreq(self, n, d=1):
        # An implementation of numpy.fft.fftfreq using Tensorflow.

        results = tf.zeros(n, dtype=tf.int32)
        N = tf.convert_to_tensor((n - 1) / 2 + 1)
        results = tf.concat(
            [tf.range(N, dtype=tf.int32), tf.range(-(n // 2), 0, dtype=tf.int32)], 0
        )
        return results / (n * d)

    def apply_fftfreq(self, h, w, z):
        # Applying the fourier transform across 3 dimensions.

        fx = self.fftfreq(w)[: w // 2 + 1 + w % 2]

        fy = self.fftfreq(h)
        fy = tf.expand_dims(fy, -1)
        fy = tf.expand_dims(fy, -1)

        fz = self.fftfreq(z)[:, None]
        return tf.math.sqrt(fx * fx + fy * fy + fz * fz)

    def get_spectrum(self, freqs, decay_power, ch, h, w, z):
        # Function to apply a low pass filter by decaying its high frequency components.

        scale = tf.ones(1) / tf.cast(
            tf.math.maximum(freqs, tf.convert_to_tensor([1 / tf.reduce_max([w, h, z])]))
            ** decay_power,
            tf.float32,
        )

        param_size = tf.concat(
            [tf.constant([ch]), tf.shape(freqs), tf.constant([2])], 0
        )
        param = self._random_generator.random_uniform(param_size)

        scale = tf.expand_dims(scale, -1)[None, :]

        return scale * param

    def sample_mask_from_transform(self, decay, shape, ch=1):
        # Sampling low frequency map from fourier transform.

        freqs = self.always_fftfreq(shape[0], shape[1], shape[2])
        spectrum = self.get_spectrum(freqs, decay, ch, shape[0], shape[1], shape[2])
        spectrum = tf.complex(spectrum[:, 0], spectrum[:, 1])

        mask = tf.math.real(tf.signal.irfft3d(spectrum, shape))
        mask = mask[:1, : shape[0], : shape[1], : shape[2]]

        mask = mask - tf.reduce_min(mask)
        mask = mask / tf.reduce_max(mask)
        return mask

    def binarise_mask(self, mask, lam, in_shape, max_soft=0.0):
        # create the final mask from the sampled values.

        idx = tf.argsort(tf.reshape(mask, [-1]), direction="DESCENDING")
        mask = tf.reshape(mask, [-1])
        num = tf.math.round(lam * tf.cast(tf.size(mask), tf.float32))

        eff_soft = max_soft

        max_compare = tf.math.maximum(lam, (1 - lam))
        min_compare = tf.math.minimum(lam, (1 - lam))
        eff_soft = tf.where(max_soft > max_compare, max_compare, max_soft)
        eff_soft = tf.where(max_soft > min_compare, min_compare, max_soft)

        soft = tf.cast(tf.size(mask), tf.float32) * eff_soft
        num_low = tf.cast(num - soft, tf.int32)
        num_high = tf.cast(num + soft, tf.int32)

        updates = tf.concat(
            [
                tf.ones((num_low,), tf.float32),
                tf.cast(tf.linspace(1, 0, num_high - num_low), tf.float32),
                tf.zeros((tf.size(mask) - num_high,), tf.float32),
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
                "FMix expects inputs in a dictionary with format "
                '{"images": images, "labels": labels}.'
                f"Got: inputs = {inputs}"
            )
        images, lambda_sample, permutation_order = self._fmix(images)
        if labels is not None:
            labels = self._update_labels(labels, lambda_sample, permutation_order)
            inputs["labels"] = labels
        inputs["images"] = images
        return inputs

    def _augment(self, inputs):
        raise ValueError(
            "FMix received a single image to `call`.  The layer relies on "
            "combining multiple examples, and as such will not behave as "
            "expected.  Please call the layer with 2 or more samples."
        )

    def _fmix(self, images):
        shape = tf.shape(images)
        permutation_order = tf.random.shuffle(tf.range(0, shape[0]), seed=self.seed)

        lambda_sample = FMix._sample_from_beta(self.alpha, self.alpha)

        masks = self.sample_mask_from_transform(self.decay_power, shape[:-1])
        masks = self.binarise_mask(masks, lambda_sample, shape[:-1], self.max_soft)
        masks = tf.expand_dims(masks, -1)
        fmix_images = tf.gather(images, permutation_order)

        images = masks * images + (1.0 - masks) * fmix_images

        return images, lambda_sample, permutation_order

    def _update_labels(self, labels, lambda_sample, permutation_order):
        labels_for_fmix = tf.gather(labels, permutation_order)

        labels = lambda_sample * labels + (1.0 - lambda_sample) * labels_for_fmix
        return labels

    def get_config(self):
        config = {
            "alpha": self.alpha,
            "decay_power": self.decay_power,
            "max_soft": self.max_soft,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
