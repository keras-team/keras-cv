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

from keras_cv.src.backend import keras


def scale_loss_for_distribution(loss_value):
    """Scales and returns the given loss value by the number of replicas."""
    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    if num_replicas > 1:
        loss_value *= 1.0 / num_replicas
    return loss_value


def convert_inputs_to_tf_dataset(
    x=None, y=None, sample_weight=None, batch_size=None
):
    if sample_weight is not None:
        raise ValueError(
            "Contrastive trainers do not yet support `sample_weight`."
        )

    if isinstance(x, tf.data.Dataset):
        if y is not None or batch_size is not None:
            raise ValueError(
                "When `x` is a `tf.data.Dataset`, please do not "
                "provide a value for `y` or `batch_size`. "
                "Got `y={y}`, `batch_size={batch_size}`."
            )
        return x

    # batch_size defaults to 32, as it does in fit().
    batch_size = batch_size or 32
    # Parse inputs
    inputs = x
    if y is not None:
        inputs = (x, y)

    # Construct tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    return dataset


def get_feature_extractor(model, layer_names, output_keys=None):
    """Create a feature extractor model with augmented output.

    This method produces a new `keras.Model` with the same input signature
    as the source but with the layers in `layer_names` as the output.
    This is useful for downstream tasks that require more output than the
    final layer of the backbone.

    Args:
        model: keras.Model. The source model.
        layer_names: list of strings. Names of layers to include in the
            output signature.
        output_keys: optional, list of strings. Key to use for each layer in
            the model's output dictionary.

    Returns:
        `keras.Model` which has dict as outputs.
    """

    if not output_keys:
        output_keys = layer_names
    items = zip(output_keys, layer_names)
    outputs = {key: model.get_layer(name).output for key, name in items}
    return keras.Model(inputs=model.inputs, outputs=outputs)
