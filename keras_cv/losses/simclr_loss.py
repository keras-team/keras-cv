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
from tensorflow import keras


class SimCLRLoss(tf.keras.losses.Loss):
    """Implements SimCLR Cosine Similarity loss.

    SimCLR loss is used for contrastive self-supervised learning.

    Args:
        temperature: a float value between 0 and 1, used as a scaling factor for cosine similarity.

    References:
        - [SimCLR paper](https://arxiv.org/pdf/2002.05709)
    """

    def __init__(self, temperature, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def call(self, projections_1, projections_2):
        """Computes SimCLR loss for a pair of projections in a contrastive learning trainer.

        Note that unlike most loss functions, this should not be called with y_true and y_pred,
        but with two unlabeled projections. It can otherwise be treated as a normal loss function.

        Args:
            projections_1: a tensor with the output of the first projection model in a contrastive learning trainer
            projections_2: a tensor with the output of the second projection model in a contrastive learning trainer

        Returns:
            A tensor with the SimCLR loss computed from the input projections
        """
        # Compute the dot product of the L2 norms of the projections
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (
            tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        )

        # Produce artificial labels, 1 for each image in the batch.
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)

        # The similarities are used as logits for cross-entropy against the artificial labels.
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2

    def get_config(self):
        config = super().get_config()
        config.update({"temperature": self.temperature})
        return config
