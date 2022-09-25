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
"""Keras implementation of StableDiffusion.

Credits:

- Original implementation: https://github.com/CompVis/stable-diffusion
- Initial TF/Keras port: https://github.com/divamgupta/stable-diffusion-tensorflow

The current implementation is a rewrite of the initial TF/Keras port by Divam Gupta.
"""

import math

import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras_cv.models.generative.stable_diffusion.clip_tokenizer import SimpleTokenizer
from keras_cv.models.generative.stable_diffusion.constants import _ALPHAS_CUMPROD
from keras_cv.models.generative.stable_diffusion.constants import _UNCONDITIONAL_TOKENS
from keras_cv.models.generative.stable_diffusion.decoder import Decoder
from keras_cv.models.generative.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.generative.stable_diffusion.text_encoder import TextEncoder

MAX_PROMPT_LENGTH = 77


class StableDiffusion:
    """Keras implementation of Stable Diffusion.

    Stable Diffusion is a powerful image generation model that can be used,
    among other things, to generate pictures according to a short text description
    (called a "prompt").

    Arguments:
        img_height: Height of the images to generate, in pixel. Note that only
            multiples of 128 are supported; the value provided will be rounded
            to the nearest valid value. Default: 512.
        img_width: Width of the images to generate, in pixel. Note that only
            multiples of 128 are supported; the value provided will be rounded
            to the nearest valid value. Default: 512.
        jit_compile: Whether to compile the underlying models to XLA.
            This can lead to a significant speedup on some systems. Default: False.

    Example:

    ```python
    from keras_cv.models import StableDiffusion
    from PIL import Image

    model = StableDiffusion(img_height=512, img_width=512, jit_compile=True)
    img = model.text_to_image(
        prompt="A beautiful horse running through a field",
        batch_size=1,  # How many images to generate at once
        num_steps=25,  # Number of iterations (controls image quality)
        seed=123,  # Set this to always get the same image from the same prompt
    )
    Image.fromarray(img[0]).save("horse.png")
    print("saved at horse.png")
    ```

    References:

    - [About Stable Diffusion](https://stability.ai/blog/stable-diffusion-announcement)
    - [Original implementation](https://github.com/CompVis/stable-diffusion)
    """

    def __init__(self, img_height=512, img_width=512, jit_compile=False):
        # UNet requires multiples of 2**7 = 128
        img_height = round(img_height / 128) * 128
        img_width = round(img_width / 128) * 128
        self.img_height = img_height
        self.img_width = img_width
        self.tokenizer = SimpleTokenizer()

        # Create models
        self.text_encoder = TextEncoder(MAX_PROMPT_LENGTH)
        self.diffusion_model = DiffusionModel(img_height, img_width, MAX_PROMPT_LENGTH)
        self.decoder = Decoder(img_height, img_width)
        if jit_compile:
            self.text_encoder.compile(jit_compile=True)
            self.diffusion_model.compile(jit_compile=True)
            self.decoder.compile(jit_compile=True)

        # Load weights
        text_encoder_weights_fpath = keras.utils.get_file(
            origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_encoder.h5",
            file_hash="4789e63e07c0e54d6a34a29b45ce81ece27060c499a709d556c7755b42bb0dc4",
        )
        diffusion_model_weights_fpath = keras.utils.get_file(
            origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_diffusion_model.h5",
            file_hash="8799ff9763de13d7f30a683d653018e114ed24a6a819667da4f5ee10f9e805fe",
        )
        decoder_weights_fpath = keras.utils.get_file(
            origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_decoder.h5",
            file_hash="ad350a65cc8bc4a80c8103367e039a3329b4231c2469a1093869a345f55b1962",
        )
        self.text_encoder.load_weights(text_encoder_weights_fpath)
        self.diffusion_model.load_weights(diffusion_model_weights_fpath)
        self.decoder.load_weights(decoder_weights_fpath)

    def text_to_image(
        self,
        prompt,
        batch_size=1,
        num_steps=25,
        unconditional_guidance_scale=7.5,
        seed=None,
    ):
        # Tokenize prompt (i.e. starting context)
        inputs = self.tokenizer.encode(prompt)
        if len(inputs) > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"Prompt is too long (should be <= {MAX_PROMPT_LENGTH} tokens)"
            )
        phrase = inputs + [49407] * (MAX_PROMPT_LENGTH - len(inputs))
        phrase = tf.convert_to_tensor([phrase], dtype=tf.int32)
        phrase = tf.repeat(phrase, batch_size, axis=0)

        # Encode prompt tokens + positions into a "context" vector
        pos_ids = tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)
        pos_ids = tf.repeat(pos_ids, batch_size, axis=0)
        context = self.text_encoder.predict_on_batch([phrase, pos_ids])

        # Encode unconditional tokens + positions as "unconditional context"
        unconditional_tokens = tf.convert_to_tensor(
            [_UNCONDITIONAL_TOKENS], dtype=tf.int32
        )
        self.unconditional_tokens = tf.repeat(unconditional_tokens, batch_size, axis=0)
        unconditional_context = self.text_encoder.predict_on_batch(
            [self.unconditional_tokens, pos_ids]
        )

        # Iterative reverse diffusion stage
        timesteps = tf.range(1, 1000, 1000 // num_steps)
        latent, alphas, alphas_prev = self._get_initial_parameters(
            timesteps, batch_size, seed
        )
        progbar = keras.utils.Progbar(len(timesteps))
        iteration = 0
        for index, timestep in list(enumerate(timesteps))[::-1]:
            latent_prev = latent  # Set aside the previous latent vector
            t_emb = self._get_timestep_embedding(timestep, batch_size)
            unconditional_latent = self.diffusion_model.predict_on_batch(
                [latent, t_emb, unconditional_context]
            )
            latent = self.diffusion_model.predict_on_batch([latent, t_emb, context])
            latent = unconditional_latent + unconditional_guidance_scale * (
                latent - unconditional_latent
            )
            a_t, a_prev = alphas[index], alphas_prev[index]
            pred_x0 = (latent_prev - math.sqrt(1 - a_t) * latent) / math.sqrt(a_t)
            latent = latent * math.sqrt(1.0 - a_prev) + math.sqrt(a_prev) * pred_x0
            iteration += 1
            progbar.update(iteration)

        # Decoding stage
        decoded = self.decoder.predict_on_batch(latent)
        decoded = ((decoded + 1) / 2) * 255
        return np.clip(decoded, 0, 255).astype("uint8")

    def _get_timestep_embedding(self, timestep, batch_size, dim=320, max_period=10000):
        half = dim // 2
        freqs = tf.math.exp(
            -math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half
        )
        args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
        embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
        embedding = tf.reshape(embedding, [1, -1])
        return tf.repeat(embedding, batch_size, axis=0)

    def _get_initial_parameters(self, timesteps, batch_size, seed=None):
        alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
        alphas_prev = [1.0] + alphas[:-1]
        noise = tf.random.normal(
            (batch_size, self.img_height // 8, self.img_width // 8, 4), seed=seed
        )
        return noise, alphas, alphas_prev
