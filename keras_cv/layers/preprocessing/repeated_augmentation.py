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
from tensorflow import keras

from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class RepeatedAugmentation(BaseImageAugmentationLayer):
    """RepeatedAugmentation augments each image in a batch multiple times.

        This technique exists to emulate the behavior of stochastic gradient descent within
        the context of mini-batch gradient descent.  When training large vision models,
        choosing a large batch size can introduce too much noise into aggregated gradients
        causing the overall batch's gradients to be less effective than gradients produced
        using smaller gradients.  RepeatedAugmentation handles this by re-using the same
        image multiple times within a batch creating correlated samples.

        Args:
            augmenters: the augmenters to use to augment the image
            shuffle: whether or not to shuffle the result.  Essential when using an
                asynchronous distribution strategy such as ParameterServerStrategy.

        Usage:
        ```python
        repeated_augment = cv_layers.RepeatedAugmentation(
            augmenters=[
                cv_layers.RandAugment(value_range=(0, 255)),
                cv_layers.RandomFlip(),
            ]
        )
        inputs = {
            "images": tf.ones((8, 512, 512, 3)),
            "labels": tf.ones((8,)),
        }
        outputs = repeated_augment(inputs)
        ```

        References:
        - [DEIT implementaton](https://github.com/facebookresearch/deit/blob/ee8893c8063f6937fec7096e47ba324c206e22b9/samplers.py#L8
    )
        - [Original publication](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hoffer_Augment_Your_Batch_Improving_Generalization_Through_Instance_Repetition_CVPR_2020_paper.pdf)

    """

    def __init__(self, augmenters, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.augmenters = augmenters
        self.shuffle = shuffle

    def _batch_augment(self, inputs):
        images = inputs.get("images", None)
        labels = inputs.get("labels", None)

        if sorted(inputs.keys()) != ["images", "labels"]:
            raise ValueError(
                "RepeatedAugmentation() does not yet support tasks other than "
                "classification."
            )

        if images is None or labels is None:
            raise ValueError(
                "RepeatedAugmentation expects inputs in a dictionary with format "
                '{"images": images, "labels": labels}.'
                f"Got: inputs = {inputs}"
            )

        image_results = []
        labels_results = []

        for augmenter in self.augmenters:
            target = augmenter(inputs)
            image_results.append(target["images"])
            labels_results.append(target["labels"])

        image_results = tf.concat(image_results, axis=0)
        labels_results = tf.concat(labels_results, axis=0)

        result = {"images": image_results, "labels": labels_results}

        if not self.shuffle:
            return result
        return self.shuffle_outputs(self, result)

    def shuffle_outputs(self, result):
        indices = tf.range(start=0, limit=tf.shape(result["images"])[0], dtype=tf.int32)
        indices = tf.random.shuffle(indices)
        result["images"] = tf.gather(result["images"], indices)
        result["labels"] = tf.gather(result["labels"], indices)
        return result

    def _augment(self, inputs):
        raise ValueError(
            "RepeatedAugmentation() only works in batched mode.  If "
            "you would like to create batches from a single image, use "
            "`x = tf.expand_dims(x, axis=0)` on your input images and labels."
        )

    def get_config(self):
        config = super().get_config()
        config.update({"augmenters": self.augmenters, "shuffle": self.shuffle})
        return config
