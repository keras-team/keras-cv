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

from keras_cv.api_export import keras_cv_export
from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.layers.detectron2_layers import MLP
from keras_cv.layers.serializable_sequential import SerializableSequential


@keras_cv_export("keras_cv.models.SAMMaskDecoder")
class SAMMaskDecoder(keras.models.Model):
    """Mask decoder for the Segment Anything Model (SAM).

    This lightweight module efficiently maps the image embedding and a set of
    prompt embeddings to an output mask. Before applying the transformer
    decoder, the layer first inserts into the set of prompt embeddings a
    learned output token embedding that will be used at the decoder's output.
    For simplicity, these embeddings (not including the image embedding) are
    collectively called "tokens".

    The image embeddings, positional image embeddings, and tokens are passed
    through a transformer decoder. After running the decoder, the layer
    upsamples the updated image embedding by 4x with two transposed
    convolutional layers (now it's downscaled 4x relative to the input
    image). Then, the tokens attend once more to the image embedding and
    the updated output token embedding are passed to a small 3-layer MLP that
    outputs a vector matching the channel dimension of the upscaled image
    embedding. Finally, a mask is predicted with a spatially point-wise
    product between the upscaled image embedding and the MLP's output.

    Args:
        transformer_dim (int): The number of input features to the transformer
            decoder.
        transformer (keras.layers.Layer): A transformer decoder.
        num_multimask_outputs (int): Number of multimask outputs. The model
            would generate these many extra masks when `multimask_output` is
            `True`.
        iou_head_depth (int): The depth of the dense net used to predict the
            IoU confidence score.
        iou_head_hidden_dim (int): The number of units in the hidden layers
            used in the dense net to predict the IoU confidence score.
        activation (str, optional): Activation to use in the mask upscaler
            network. Defaults to "gelu".

    References:
        - [Segment Anything](https://arxiv.org/abs/2304.02643)
    """

    def __init__(
        self,
        transformer_dim,
        transformer,
        num_multimask_outputs,
        iou_head_depth,
        iou_head_hidden_dim,
        activation="gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim
        self.activation = activation

        self.iou_token = keras.layers.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = keras.layers.Embedding(
            self.num_mask_tokens, transformer_dim
        )

        self.output_upscaling = SerializableSequential(
            [
                keras.layers.Conv2DTranspose(
                    transformer_dim // 4, kernel_size=2, strides=2
                ),
                keras.layers.LayerNormalization(epsilon=1e-6),
                keras.layers.Activation(activation),
                keras.layers.Conv2DTranspose(
                    transformer_dim // 8, kernel_size=2, strides=2
                ),
                keras.layers.Activation(activation),
            ]
        )

        self.output_hypernetworks_mlps = [
            MLP(transformer_dim, transformer_dim // 8, 3)
            for _ in range(self.num_mask_tokens)
        ]

        self.iou_prediction_head = MLP(
            iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        self.iou_token.build(None)
        self.mask_tokens.build(None)

        self.output_upscaling.build([None, None, None, self.transformer_dim])

        for mlp in self.output_hypernetworks_mlps:
            mlp.build([None, self.transformer_dim])

        self.iou_prediction_head.build([None, self.transformer_dim])

        self.built = True

    def call(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output,
    ):
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        if multimask_output:
            return masks[:, 1:, :, :], iou_pred[:, 1:]
        return masks[:, :1, :, :], iou_pred[:, :1]

    def predict_masks(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
    ):
        output_tokens = ops.concatenate(
            [self.iou_token.weights[0], self.mask_tokens.weights[0]], axis=0
        )
        output_tokens = ops.broadcast_to(
            output_tokens[None, ...],
            shape=(
                sparse_prompt_embeddings.shape[0],
                output_tokens.shape[0],
                output_tokens.shape[1],
            ),
        )
        tokens = ops.concatenate(
            [output_tokens, sparse_prompt_embeddings], axis=1
        )

        source = ops.broadcast_to(
            image_embeddings,
            shape=(
                tokens.shape[0],
                image_embeddings.shape[1],
                image_embeddings.shape[2],
                image_embeddings.shape[3],
            ),
        )
        source = source + dense_prompt_embeddings
        positional_source = ops.broadcast_to(
            image_pe,
            shape=(
                tokens.shape[0],
                image_embeddings.shape[1],
                image_embeddings.shape[2],
                image_embeddings.shape[3],
            ),
        )
        B, H, W, C = source.shape

        hidden_state, source = self.transformer(
            source, positional_source, tokens
        )
        iou_token_out = hidden_state[:, 0, :]
        mask_tokens_out = hidden_state[:, 1 : (1 + self.num_mask_tokens), :]

        source = ops.reshape(source, (B, H, W, C))
        upscaled_embeddings = self.output_upscaling(source)
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = ops.stack(hyper_in_list, axis=1)
        B, H, W, C = upscaled_embeddings.shape
        upscaled_embeddings = ops.reshape(
            ops.transpose(upscaled_embeddings, axes=(0, 3, 1, 2)),
            (B, C, H * W),
        )
        masks = ops.reshape(
            hyper_in @ upscaled_embeddings, (B, self.num_mask_tokens, H, W)
        )

        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "transformer_dim": self.transformer_dim,
                "transformer": keras.saving.serialize_keras_object(
                    self.transformer
                ),
                "num_multimask_outputs": self.num_multimask_outputs,
                "iou_head_depth": self.iou_head_depth,
                "iou_head_hidden_dim": self.iou_head_hidden_dim,
                "activation": self.activation,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config.update(
            {"transformer": keras.layers.deserialize(config["transformer"])}
        )
        return super().from_config(config)
