# Copyright 2023 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility methods used to implement Prompt-to-Prompt paper in TensorFlow.

Reference:

- "Prompt-to-Prompt Image Editing with Cross-Attention Control."
  Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, Daniel Cohen-Or.
  https://arxiv.org/abs/2208.01626

Credits:

- Official implementation of the paper: https://github.com/google/prompt-to-prompt
"""

from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

import keras_cv.models.stable_diffusion.seq_aligner as seq_aligner
from keras_cv.models.stable_diffusion.diffusion_model import td_dot


def rename_cross_attn_layers(diff_model: tf.keras.Model):
    """Add suffix to the cross attention layers.

    This becomes useful when using the prompt editing method to save the
    attention maps and manipulate the control variables.

    Args:
        diff_model (tf.keras.Model): The diffusion model.
    """
    cross_attention_count = 0
    for submodule in diff_model.submodules:
        submodule_name = submodule.name
        if not cross_attention_count % 2 and "cross_attention" in submodule_name:
            submodule._name = f"{submodule_name}_attn1"
            cross_attention_count += 1
        elif cross_attention_count % 2 and "cross_attention" in submodule_name:
            submodule._name = f"{submodule_name}_attn2"
            cross_attention_count += 1


def update_cross_attn_mode(
    diff_model: tf.keras.Model, mode: str, attn_suffix: str = "attn"
):
    """Update the mode control variable.

    Args:
        diff_model (tf.keras.Model): The diffusion model.
        mode (str): The mode parameter can take 5 values:
                    - save: to save the attention map.
                    - use_last: to use the original cross attention maps.
                    - edit: to calculate the attention map with respect to the edited prompt.
                    - injection: to use the edited prompt cross attention maps
                    - unconditional: to perform the standard attention computations.
        attn_suffix (str): Suffix used to find the attention layer, by default "attn".
    """
    for submodule in diff_model.submodules:
        submodule_name = submodule.name
        if (
            "cross_attention" in submodule_name
            and attn_suffix in submodule_name.split("_")[-1]
        ):
            submodule.cross_attn_mode.assign(mode)


def update_attn_weights_usage(diff_model: tf.keras.Model, use: bool):
    """Update the mode control variable.

    Args:
        diff_model (tf.keras.Model): The diffusion model.
        use (bool): Whether to use the prompt weights.
    """
    for submodule in diff_model.submodules:
        submodule_name = submodule.name
        if (
            "cross_attention" in submodule_name
            and "attn2" in submodule_name.split("_")[-1]
        ):
            submodule.use_prompt_weights.assign(use)


def add_attn_weights(diff_model: tf.keras.Model, prompt_weights: np.ndarray):
    """Assign the attention weights to the diffusion model's corresponding tf.variable.

    Args:
        diff_model (tf.keras.Model): The diffusion model.
        prompt_weights (np.ndarray): Weights of the attention tokens.
    """
    for submodule in diff_model.submodules:
        submodule_name = submodule.name
        if (
            "cross_attention" in submodule_name
            and "attn2" in submodule_name.split("_")[-1]
        ):
            submodule.prompt_weights.assign(prompt_weights)


def put_mask_dif_model(
    diff_model: tf.keras.Model, mask: np.ndarray, indices: np.ndarray
):
    """Assign the diffusion model's tf.variables with the passed mask and indices.

    Args:
        diff_model (tf.keras.Model): The diffusion model.
        mask (np.ndarray): Mask of the original and edited prompt overlap.
        indices (np.ndarray): Indices of the original and edited prompt overlap.
    """
    for submodule in diff_model.submodules:
        submodule_name = submodule.name
        if (
            "cross_attention" in submodule_name
            and "attn2" in submodule_name.split("_")[-1]
        ):
            submodule.prompt_edit_mask.assign(mask)
            submodule.prompt_edit_indices.assign(indices)


def get_matching_sentence_tokens(
    prompt, prompt_edit, tokenizer
) -> Tuple[np.ndarray, np.ndarray]:
    """Create the mask and indices of the overlap between the tokens of the original \
    prompt and the edited one.

    Original code source: https://github.com/google/prompt-to-prompt

    Args:
        tokens (np.ndarray): Array of the original prompt tokens.
        tokens_edit (np.ndarray): Array of the edit prompt tokens.

    Returns:
        mask (np.ndarray): Mask of the original and edited prompt overlap.
        indices: (np.ndarray): Indices of the original and edited prompt overlap.
    """
    tokens_conditional = tokenizer.encode(prompt)
    tokens_conditional_edit = tokenizer.encode(prompt_edit)
    mask, indices = seq_aligner.get_mapper(tokens_conditional, tokens_conditional_edit)
    return mask, indices


def set_initial_tf_variables(diff_model: tf.keras.Model):
    """Create initial control variables to auxiliate the prompt editing method.

    Args:
        diff_model (tf.keras.Model): The diffusion model.
    """
    for submodule in diff_model.submodules:
        submodule_name = submodule.name
        if "cross_attention" in submodule_name:
            # Set control variables
            submodule.cross_attn_mode = tf.Variable(
                "", dtype=tf.string, trainable=False
            )
            submodule.use_prompt_weights = tf.Variable(
                False, dtype=tf.bool, trainable=False
            )
            # Set array variables
            submodule.attn_map = tf.Variable(
                [], shape=tf.TensorShape(None), dtype=tf.float32, trainable=False
            )
            submodule.prompt_edit_mask = tf.Variable(
                [], shape=tf.TensorShape(None), dtype=tf.float32, trainable=False
            )
            submodule.prompt_edit_indices = tf.Variable(
                [], shape=tf.TensorShape(None), dtype=tf.int32, trainable=False
            )
            submodule.prompt_weights = tf.Variable(
                [], shape=tf.TensorShape(None), dtype=tf.float32, trainable=False
            )


def reset_initial_tf_variables(diff_model: tf.keras.Model):
    """Reset the control variables to their default values.

    Args:
        diff_model (tf.keras.Model): The diffusion model.
    """
    for submodule in diff_model.submodules:
        submodule_name = submodule.name
        if "cross_attention" in submodule_name:
            # Reset control variables
            submodule.cross_attn_mode.assign("")
            submodule.use_prompt_weights.assign(False)
            # Reset array variables
            submodule.attn_map.assign([])
            submodule.prompt_edit_mask.assign([])
            submodule.prompt_edit_indices.assign([])
            submodule.prompt_weights.assign([])


def overwrite_forward_call(diff_model: tf.keras.Model):
    """Update the attention forward pass with a custom call method.

    Args:
        diff_model (tf.keras.Model): The diffusion model.
    """
    for submodule in diff_model.submodules:
        submodule_name = submodule.name
        if "cross_attention" in submodule_name:
            # Overwrite forward pass method
            submodule.call = call_attn_edit.__get__(submodule)


def call_attn_edit(self, inputs):
    """Implmentation of the custom attention forward pass used in the paper's method."""
    inputs, context = inputs
    context = inputs if context is None else context
    q, k, v = self.to_q(inputs), self.to_k(context), self.to_v(context)
    q = tf.reshape(q, (-1, inputs.shape[1], self.num_heads, self.head_size))
    k = tf.reshape(k, (-1, context.shape[1], self.num_heads, self.head_size))
    v = tf.reshape(v, (-1, context.shape[1], self.num_heads, self.head_size))

    q = tf.transpose(q, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)
    k = tf.transpose(k, (0, 2, 3, 1))  # (bs, num_heads, head_size, time)
    v = tf.transpose(v, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)

    score = td_dot(q, k) * self.scale
    weights = keras.activations.softmax(score)  # (bs, num_heads, time, time)

    # Method: Prompt Refinement
    if tf.equal(self.cross_attn_mode, "edit") and tf.not_equal(
        tf.size(self.prompt_edit_mask), 0
    ):  # not empty
        weights_masked = tf.gather(self.attn_map, self.prompt_edit_indices, axis=-1)
        edit_weights = weights_masked * self.prompt_edit_mask + weights * (
            1 - self.prompt_edit_mask
        )
        weights = tf.reshape(edit_weights, shape=tf.shape(weights))

    # Use the attention from the original prompt (M_t)
    if tf.equal(self.cross_attn_mode, "use_last"):
        weights = tf.reshape(self.attn_map, shape=tf.shape(weights))

    # Save attention
    if tf.equal(self.cross_attn_mode, "save"):
        self.attn_map.assign(weights)

    # Method: Attention Re–weighting
    if tf.equal(self.use_prompt_weights, True) and tf.not_equal(
        tf.size(self.prompt_weights), 0
    ):
        attn_map_weighted = weights * self.prompt_weights
        weights = tf.reshape(attn_map_weighted, shape=tf.shape(weights))

    attn = td_dot(weights, v)
    attn = tf.transpose(attn, (0, 2, 1, 3))  # (bs, time, num_heads, head_size)
    out = tf.reshape(attn, (-1, inputs.shape[1], self.num_heads * self.head_size))
    return self.out_proj(out)
