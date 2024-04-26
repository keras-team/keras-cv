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
"""Image classifier model using pooling and dense layers."""

import copy

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.models.backbones.backbone_presets import backbone_presets
from keras_cv.src.models.backbones.backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.src.models.classification.image_classifier_presets import (
    classifier_presets,
)
from keras_cv.src.models.task import Task
from keras_cv.src.utils.python_utils import classproperty


@keras_cv_export(
    [
        "keras_cv.models.ImageClassifier",
        "keras_cv.models.classification.ImageClassifier",
    ]
)
class ImageClassifier(Task):
    """Image classifier with pooling and dense layer prediction head.

    Args:
        backbone: `keras.Model` instance, the backbone architecture of the
            classifier called on the inputs. Pooling will be called on the last
            dimension of the backbone output.
        num_classes: int, number of classes to predict.
        pooling: str, type of pooling layer. Must be one of "avg", "max".
        activation: Optional `str` or callable, defaults to "softmax". The
            activation function to use on the Dense layer. Set `activation=None`
            to return the output logits.

    Example:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Pretrained classifier (e.g., for imagenet categories)
    model = keras_cv.models.ImageClassifier.from_preset(
        "resnet50_v2_imagenet_classifier",
    )
    output = model(input_data)

    # Pretrained backbone
    backbone = keras_cv.models.ResNet50V2Backbone.from_preset(
        "resnet50_v2_imagenet",
    )
    model = keras_cv.models.ImageClassifier(
        backbone=backbone,
        num_classes=4,
    )
    output = model(input_data)

    # Randomly initialized backbone with a custom config
    model = keras_cv.models.ImageClassifier(
        backbone=keras_cv.models.ResNet50V2Backbone(),
        num_classes=4,
    )
    output = model(input_data)
    ```
    """

    def __init__(
        self,
        backbone,
        num_classes,
        pooling="avg",
        activation="softmax",
        **kwargs,
    ):
        if pooling == "avg":
            pooling_layer = keras.layers.GlobalAveragePooling2D(name="avg_pool")
        elif pooling == "max":
            pooling_layer = keras.layers.GlobalMaxPooling2D(name="max_pool")
        else:
            raise ValueError(
                f'`pooling` must be one of "avg", "max". Received: {pooling}.'
            )
        inputs = backbone.input
        x = backbone(inputs)
        x = pooling_layer(x)
        outputs = keras.layers.Dense(
            num_classes,
            activation=activation,
            name="predictions",
        )(x)

        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )
        # All references to `self` below this line
        self.backbone = backbone
        self.num_classes = num_classes
        self.pooling = pooling
        self.activation = activation

    def get_config(self):
        # Backbone serialized in `super`
        config = super().get_config()
        config.update(
            {
                "backbone": keras.layers.serialize(self.backbone),
                "num_classes": self.num_classes,
                "pooling": self.pooling,
                "activation": self.activation,
            }
        )
        return config

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return copy.deepcopy({**backbone_presets, **classifier_presets})

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return copy.deepcopy(
            {**backbone_presets_with_weights, **classifier_presets}
        )

    @classproperty
    def backbone_presets(cls):
        """Dictionary of preset names and configurations of compatible
        backbones."""
        return copy.deepcopy(backbone_presets)
