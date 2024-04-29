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

- Original implementation:
  https://github.com/CompVis/stable-diffusion
- Initial TF/Keras port:
  https://github.com/divamgupta/stable-diffusion-tensorflow

The current implementation is a rewrite of the initial TF/Keras port by
Divam Gupta.
"""

import math

import numpy as np

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.backend import random
from keras_cv.src.models.stable_diffusion.clip_tokenizer import SimpleTokenizer
from keras_cv.src.models.stable_diffusion.constants import _ALPHAS_CUMPROD
from keras_cv.src.models.stable_diffusion.constants import _UNCONDITIONAL_TOKENS
from keras_cv.src.models.stable_diffusion.decoder import Decoder
from keras_cv.src.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.src.models.stable_diffusion.diffusion_model import (
    DiffusionModelV2,
)
from keras_cv.src.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.src.models.stable_diffusion.text_encoder import TextEncoder
from keras_cv.src.models.stable_diffusion.text_encoder import TextEncoderV2

MAX_PROMPT_LENGTH = 77


class StableDiffusionBase:
    """Base class for stable diffusion and stable diffusion v2 model."""

    def __init__(
        self,
        img_height=512,
        img_width=512,
        jit_compile=True,
    ):
        # UNet requires multiples of 2**7 = 128
        img_height = round(img_height / 128) * 128
        img_width = round(img_width / 128) * 128
        self.img_height = img_height
        self.img_width = img_width

        # lazy initialize the component models and the tokenizer
        self._image_encoder = None
        self._text_encoder = None
        self._diffusion_model = None
        self._decoder = None
        self._tokenizer = None

        self.jit_compile = jit_compile

    def text_to_image(
        self,
        prompt,
        negative_prompt=None,
        batch_size=1,
        num_steps=50,
        unconditional_guidance_scale=7.5,
        seed=None,
    ):
        encoded_text = self.encode_text(prompt)

        return self.generate_image(
            encoded_text,
            negative_prompt=negative_prompt,
            batch_size=batch_size,
            num_steps=num_steps,
            unconditional_guidance_scale=unconditional_guidance_scale,
            seed=seed,
        )

    def encode_text(self, prompt):
        """Encodes a prompt into a latent text encoding.

        The encoding produced by this method should be used as the
        `encoded_text` parameter of `StableDiffusion.generate_image`. Encoding
        text separately from generating an image can be used to arbitrarily
        modify the text encoding prior to image generation, e.g. for walking
        between two prompts.

        Args:
            prompt: a string to encode, must be 77 tokens or shorter.

        Example:

        ```python
        from keras_cv.src.models import StableDiffusion

        model = StableDiffusion(img_height=512, img_width=512, jit_compile=True)
        encoded_text  = model.encode_text("Tacos at dawn")
        img = model.generate_image(encoded_text)
        ```
        """
        # Tokenize prompt (i.e. starting context)
        inputs = self.tokenizer.encode(prompt)
        if len(inputs) > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"Prompt is too long (should be <= {MAX_PROMPT_LENGTH} tokens)"
            )
        phrase = inputs + [49407] * (MAX_PROMPT_LENGTH - len(inputs))
        phrase = ops.convert_to_tensor([phrase], dtype="int32")

        context = self.text_encoder.predict_on_batch(
            {"tokens": phrase, "positions": self._get_pos_ids()}
        )

        return context

    def generate_image(
        self,
        encoded_text,
        negative_prompt=None,
        batch_size=1,
        num_steps=50,
        unconditional_guidance_scale=7.5,
        diffusion_noise=None,
        seed=None,
    ):
        """Generates an image based on encoded text.

        The encoding passed to this method should be derived from
        `StableDiffusion.encode_text`.

        Args:
            encoded_text: Tensor of shape (`batch_size`, 77, 768), or a Tensor
                of shape (77, 768). When the batch axis is omitted, the same
                encoded text will be used to produce every generated image.
            batch_size: int, number of images to generate, defaults to 1.
            negative_prompt: a string containing information to negatively guide
                the image generation (e.g. by removing or altering certain
                aspects of the generated image), defaults to None.
            num_steps: int, number of diffusion steps (controls image quality),
                defaults to 50.
            unconditional_guidance_scale: float, controlling how closely the
                image should adhere to the prompt. Larger values result in more
                closely adhering to the prompt, but will make the image noisier.
                Defaults to 7.5.
            diffusion_noise: Tensor of shape (`batch_size`, img_height // 8,
                img_width // 8, 4), or a Tensor of shape (img_height // 8,
                img_width // 8, 4). Optional custom noise to seed the diffusion
                process. When the batch axis is omitted, the same noise will be
                used to seed diffusion for every generated image.
            seed: integer which is used to seed the random generation of
                diffusion noise, only to be specified if `diffusion_noise` is
                None.

        Example:

        ```python
        from keras_cv.src.models import StableDiffusion
        from keras_core import ops

        batch_size = 8
        model = StableDiffusion(img_height=512, img_width=512, jit_compile=True)
        e_tacos = model.encode_text("Tacos at dawn")
        e_watermelons = model.encode_text("Watermelons at dusk")

        e_interpolated = ops.linspace(e_tacos, e_watermelons, batch_size)
        images = model.generate_image(e_interpolated, batch_size=batch_size)
        ```
        """
        if diffusion_noise is not None and seed is not None:
            raise ValueError(
                "`diffusion_noise` and `seed` should not both be passed to "
                "`generate_image`. `seed` is only used to generate diffusion "
                "noise when it's not already user-specified."
            )

        context = self._expand_tensor(encoded_text, batch_size)

        if negative_prompt is None:
            unconditional_context = ops.repeat(
                self._get_unconditional_context(), batch_size, axis=0
            )
        else:
            unconditional_context = self.encode_text(negative_prompt)
            unconditional_context = self._expand_tensor(
                unconditional_context, batch_size
            )

        if diffusion_noise is not None:
            diffusion_noise = ops.squeeze(diffusion_noise)
            if len(ops.shape(diffusion_noise)) == 3:
                diffusion_noise = ops.repeat(
                    ops.expand_dims(diffusion_noise, axis=0), batch_size, axis=0
                )
            latent = diffusion_noise
        else:
            latent = self._get_initial_diffusion_noise(batch_size, seed)

        # Iterative reverse diffusion stage
        num_timesteps = 1000
        ratio = (
            (num_timesteps - 1) / (num_steps - 1)
            if num_steps > 1
            else num_timesteps
        )
        timesteps = (np.arange(0, num_steps) * ratio).round().astype(np.int64)

        alphas, alphas_prev = self._get_initial_alphas(timesteps)
        progbar = keras.utils.Progbar(len(timesteps))
        iteration = 0
        for index, timestep in list(enumerate(timesteps))[::-1]:
            latent_prev = latent  # Set aside the previous latent vector
            t_emb = self._get_timestep_embedding(timestep, batch_size)
            unconditional_latent = self.diffusion_model.predict_on_batch(
                {
                    "latent": latent,
                    "timestep_embedding": t_emb,
                    "context": unconditional_context,
                }
            )
            latent = self.diffusion_model.predict_on_batch(
                {
                    "latent": latent,
                    "timestep_embedding": t_emb,
                    "context": context,
                }
            )
            latent = ops.array(
                unconditional_latent
                + unconditional_guidance_scale * (latent - unconditional_latent)
            )
            a_t, a_prev = alphas[index], alphas_prev[index]
            # Keras backend array need to cast explicitly
            target_dtype = latent_prev.dtype
            latent = ops.cast(latent, target_dtype)
            pred_x0 = (latent_prev - math.sqrt(1 - a_t) * latent) / math.sqrt(
                a_t
            )
            latent = (
                ops.array(latent) * math.sqrt(1.0 - a_prev)
                + math.sqrt(a_prev) * pred_x0
            )
            iteration += 1
            progbar.update(iteration)

        # Decoding stage
        decoded = self.decoder.predict_on_batch(latent)
        decoded = ((decoded + 1) / 2) * 255
        return np.clip(decoded, 0, 255).astype("uint8")

    def _get_unconditional_context(self):
        unconditional_tokens = ops.convert_to_tensor(
            [_UNCONDITIONAL_TOKENS],
            dtype="int32",
        )
        unconditional_context = self.text_encoder.predict_on_batch(
            {"tokens": unconditional_tokens, "positions": self._get_pos_ids()}
        )

        return unconditional_context

    def _expand_tensor(self, text_embedding, batch_size):
        """Extends a tensor by repeating it to fit the shape of the given batch
        size."""
        text_embedding = ops.squeeze(text_embedding)
        if len(text_embedding.shape) == 2:
            text_embedding = ops.repeat(
                ops.expand_dims(text_embedding, axis=0), batch_size, axis=0
            )
        return text_embedding

    @property
    def image_encoder(self):
        """image_encoder returns the VAE Encoder with pretrained weights.

        Example:
        ```python
        sd = keras_cv.models.StableDiffusion()
        my_image = np.ones((512, 512, 3))
        latent_representation = sd.image_encoder.predict(my_image)
        ```
        """
        if self._image_encoder is None:
            self._image_encoder = ImageEncoder()
            if self.jit_compile:
                self._image_encoder.compile(jit_compile=True)
        return self._image_encoder

    @property
    def text_encoder(self):
        pass

    @property
    def diffusion_model(self):
        pass

    @property
    def decoder(self):
        """decoder returns the diffusion image decoder model with pretrained
        weights. Can be overriden for tasks where the decoder needs to be
        modified.
        """
        if self._decoder is None:
            self._decoder = Decoder(self.img_height, self.img_width)
            if self.jit_compile:
                self._decoder.compile(jit_compile=True)
        return self._decoder

    @property
    def tokenizer(self):
        """tokenizer returns the tokenizer used for text inputs.
        Can be overriden for tasks like textual inversion where the tokenizer
        needs to be modified.
        """
        if self._tokenizer is None:
            self._tokenizer = SimpleTokenizer()
        return self._tokenizer

    def _get_timestep_embedding(
        self, timestep, batch_size, dim=320, max_period=10000
    ):
        half = dim // 2
        range = ops.cast(ops.arange(0, half), "float32")
        freqs = ops.exp(-math.log(max_period) * range / half)
        args = ops.convert_to_tensor([timestep], dtype="float32") * freqs
        embedding = ops.concatenate([ops.cos(args), ops.sin(args)], 0)
        embedding = ops.reshape(embedding, [1, -1])
        return ops.repeat(embedding, batch_size, axis=0)

    def _get_initial_alphas(self, timesteps):
        alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
        alphas_prev = [1.0] + alphas[:-1]

        return alphas, alphas_prev

    def _get_initial_diffusion_noise(self, batch_size, seed):
        return random.normal(
            (batch_size, self.img_height // 8, self.img_width // 8, 4),
            seed=seed,
        )

    @staticmethod
    def _get_pos_ids():
        return ops.expand_dims(ops.arange(MAX_PROMPT_LENGTH, dtype="int32"), 0)


@keras_cv_export("keras_cv.models.StableDiffusion")
class StableDiffusion(StableDiffusionBase):
    """Keras implementation of Stable Diffusion.

    Note that the StableDiffusion API, as well as the APIs of the sub-components
    of StableDiffusion (e.g. ImageEncoder, DiffusionModel) should be considered
    unstable at this point. We do not guarantee backwards compatability for
    future changes to these APIs.

    Stable Diffusion is a powerful image generation model that can be used,
    among other things, to generate pictures according to a short text
    description (called a "prompt").

    Arguments:
        img_height: int, height of the images to generate, in pixel. Note that
            only multiples of 128 are supported; the value provided will be
            rounded to the nearest valid value. Defaults to 512.
        img_width: int, width of the images to generate, in pixel. Note that
            only multiples of 128 are supported; the value provided will be
            rounded to the nearest valid value. Defaults to 512.
        jit_compile: bool, whether to compile the underlying models to XLA.
            This can lead to a significant speedup on some systems. Defaults to
            False.

    Example:

    ```python
    from keras_cv.src.models import StableDiffusion
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
    """  # noqa: E501

    def __init__(
        self,
        img_height=512,
        img_width=512,
        jit_compile=True,
    ):
        super().__init__(img_height, img_width, jit_compile)
        print(
            "By using this model checkpoint, you acknowledge that its usage is "
            "subject to the terms of the CreativeML Open RAIL-M license at "
            "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/LICENSE"  # noqa: E501
        )

    @property
    def text_encoder(self):
        """text_encoder returns the text encoder with pretrained weights.
        Can be overriden for tasks like textual inversion where the text encoder
        needs to be modified.
        """
        if self._text_encoder is None:
            self._text_encoder = TextEncoder(MAX_PROMPT_LENGTH)
            if self.jit_compile:
                self._text_encoder.compile(jit_compile=True)
        return self._text_encoder

    @property
    def diffusion_model(self):
        """diffusion_model returns the diffusion model with pretrained weights.
        Can be overriden for tasks where the diffusion model needs to be
        modified.
        """
        if self._diffusion_model is None:
            self._diffusion_model = DiffusionModel(
                self.img_height, self.img_width, MAX_PROMPT_LENGTH
            )
            if self.jit_compile:
                self._diffusion_model.compile(jit_compile=True)
        return self._diffusion_model


@keras_cv_export("keras_cv.models.StableDiffusionV2")
class StableDiffusionV2(StableDiffusionBase):
    """Keras implementation of Stable Diffusion v2.

    Note that the StableDiffusion API, as well as the APIs of the sub-components
    of StableDiffusionV2 (e.g. ImageEncoder, DiffusionModelV2) should be
    considered unstable at this point. We do not guarantee backwards
    compatability for future changes to these APIs.

    Stable Diffusion is a powerful image generation model that can be used,
    among other things, to generate pictures according to a short text
    description (called a "prompt").

    Arguments:
        img_height: int, height of the images to generate, in pixel. Note that
            only multiples of 128 are supported; the value provided will be
            rounded to the nearest valid value. Defaults to 512.
        img_width: int, width of the images to generate, in pixel. Note that
            only multiples of 128 are supported; the value provided will be
            rounded to the nearest valid value. Defaults to 512.
        jit_compile: bool, whether to compile the underlying models to XLA.
            This can lead to a significant speedup on some systems. Defaults to
            False.
    Example:

    ```python
    from keras_cv.src.models import StableDiffusionV2
    from PIL import Image

    model = StableDiffusionV2(img_height=512, img_width=512, jit_compile=True)
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
    - [Original implementation](https://github.com/Stability-AI/stablediffusion)
    """  # noqa: E501

    def __init__(
        self,
        img_height=512,
        img_width=512,
        jit_compile=True,
    ):
        super().__init__(img_height, img_width, jit_compile)
        print(
            "By using this model checkpoint, you acknowledge that its usage is "
            "subject to the terms of the CreativeML Open RAIL++-M license at "
            "https://github.com/Stability-AI/stablediffusion/blob/main/LICENSE-MODEL"  # noqa: E501
        )

    @property
    def text_encoder(self):
        """text_encoder returns the text encoder with pretrained weights.
        Can be overriden for tasks like textual inversion where the text encoder
        needs to be modified.
        """
        if self._text_encoder is None:
            self._text_encoder = TextEncoderV2(MAX_PROMPT_LENGTH)
            if self.jit_compile:
                self._text_encoder.compile(jit_compile=True)
        return self._text_encoder

    @property
    def diffusion_model(self):
        """diffusion_model returns the diffusion model with pretrained weights.
        Can be overriden for tasks where the diffusion model needs to be
        modified.
        """
        if self._diffusion_model is None:
            self._diffusion_model = DiffusionModelV2(
                self.img_height, self.img_width, MAX_PROMPT_LENGTH
            )
            if self.jit_compile:
                self._diffusion_model.compile(jit_compile=True)
        return self._diffusion_model
