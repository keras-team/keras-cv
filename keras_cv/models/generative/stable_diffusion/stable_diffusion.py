import numpy as np
from tqdm import tqdm
import math

import tensorflow as tf
from tensorflow import keras

from keras_cv.models.generative.stable_diffusion.decoder import Decoder
from keras_cv.models.generative.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.generative.stable_diffusion.text_encoder import TextEncoder
from keras_cv.models.generative.stable_diffusion.clip_tokenizer import SimpleTokenizer
from keras_cv.models.generative.stable_diffusion.constants import (
    _UNCONDITIONAL_TOKENS,
    _ALPHAS_CUMPROD,
)

MAX_PROMPT_LENGTH = 77


class StableDiffusion:
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
        phrase = np.array(phrase)[None].astype("int32")
        phrase = np.repeat(phrase, batch_size, axis=0)

        # Encode prompt tokens + positions into a "context" vector
        pos_ids = np.array(list(range(MAX_PROMPT_LENGTH)))[None].astype("int32")
        pos_ids = np.repeat(pos_ids, batch_size, axis=0)
        context = self.text_encoder.predict_on_batch([phrase, pos_ids])

        # Encode unconditional tokens + positions as "unconditional context"
        unconditional_tokens = np.array(_UNCONDITIONAL_TOKENS)[None].astype("int32")
        unconditional_tokens = np.repeat(unconditional_tokens, batch_size, axis=0)
        self.unconditional_tokens = tf.convert_to_tensor(unconditional_tokens)
        unconditional_context = self.text_encoder.predict_on_batch(
            [self.unconditional_tokens, pos_ids]
        )

        # Iterative reverse diffusion stage
        timesteps = np.arange(1, 1000, 1000 // num_steps)
        latent, alphas, alphas_prev = self.get_initial_parameters(
            timesteps, batch_size, seed
        )
        progbar = tqdm(list(enumerate(timesteps))[::-1])
        for index, timestep in progbar:
            progbar.set_description(f"{index:3d} {timestep:3d}")
            latent_prev = latent  # Set aside the previous latent vector
            t_emb = self.get_timestep_embedding(timestep, batch_size)
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

        # Decoding stage
        decoded = self.decoder.predict_on_batch(latent)
        decoded = ((decoded + 1) / 2) * 255
        return np.clip(decoded, 0, 255).astype("uint8")

    def get_timestep_embedding(self, timestep, batch_size, dim=320, max_period=10000):
        half = dim // 2
        freqs = np.exp(
            -math.log(max_period) * np.arange(0, half, dtype="float32") / half
        )
        args = np.array([timestep]) * freqs
        embedding = np.concatenate([np.cos(args), np.sin(args)])
        embedding = tf.convert_to_tensor(embedding.reshape(1, -1))
        return np.repeat(embedding, batch_size, axis=0)

    def get_initial_parameters(self, timesteps, batch_size, seed=None):
        alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
        alphas_prev = [1.0] + alphas[:-1]
        noise = tf.random.normal(
            (batch_size, self.img_height // 8, self.img_width // 8, 4), seed=seed
        )
        return noise, alphas, alphas_prev


if __name__ == "__main__":
    # TODO: delete later, this is for testing only
    stablediff = StableDiffusion()
    img = stablediff.text_to_image("epic breathtaking crystal caves on mars")
    from PIL import Image

    Image.fromarray(img[0]).save("test.png")
    print(f"saved at test.png")
