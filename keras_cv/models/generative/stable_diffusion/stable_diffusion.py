import numpy as np
from tqdm import tqdm
import math

import tensorflow as tf
from tensorflow import keras

from decoder import Decoder
from diffusion_model import DiffusionModel
from text_encoder import TextEncoder
from clip_tokenizer import SimpleTokenizer
from constants import _UNCONDITIONAL_TOKENS, _ALPHAS_CUMPROD


class StableDiffusion:
    def __init__(self, img_height=512, img_width=512, jit_compile=False):
        self.img_height = img_height
        self.img_width = img_width
        self.tokenizer = SimpleTokenizer()

        text_encoder, diffusion_model, decoder = get_models(img_height, img_width)
        self.text_encoder = text_encoder
        self.diffusion_model = diffusion_model
        self.decoder = decoder
        if jit_compile:
            self.text_encoder.compile(jit_compile=True)
            self.diffusion_model.compile(jit_compile=True)
            self.decoder.compile(jit_compile=True)

    def text_to_image(
        self,
        prompt,
        batch_size=1,
        num_steps=25,
        unconditional_guidance_scale=7.5,
        temperature=1,
        seed=None,
    ):
        # Tokenize prompt (i.e. starting context)
        inputs = self.tokenizer.encode(prompt)
        assert len(inputs) < 77, "Prompt is too long (should be < 77 tokens)"
        phrase = inputs + [49407] * (77 - len(inputs))
        phrase = np.array(phrase)[None].astype("int32")
        phrase = np.repeat(phrase, batch_size, axis=0)

        # Encode prompt tokens (and their positions) into a "context vector"
        pos_ids = np.array(list(range(77)))[None].astype("int32")
        pos_ids = np.repeat(pos_ids, batch_size, axis=0)
        context = self.text_encoder.predict_on_batch([phrase, pos_ids])

        # Encode unconditional tokens (and their positions into an
        # "unconditional context vector"
        unconditional_tokens = np.array(_UNCONDITIONAL_TOKENS)[None].astype("int32")
        unconditional_tokens = np.repeat(unconditional_tokens, batch_size, axis=0)
        self.unconditional_tokens = tf.convert_to_tensor(unconditional_tokens)
        unconditional_context = self.text_encoder.predict_on_batch(
            [self.unconditional_tokens, pos_ids]
        )
        timesteps = np.arange(1, 1000, 1000 // num_steps)
        latent, alphas, alphas_prev = self.get_starting_parameters(
            timesteps, batch_size, seed
        )

        # Diffusion stage
        progbar = tqdm(list(enumerate(timesteps))[::-1])
        for index, timestep in progbar:
            progbar.set_description(f"{index:3d} {timestep:3d}")
            e_t = self.get_model_output(
                latent,
                timestep,
                context,
                unconditional_context,
                unconditional_guidance_scale,
                batch_size,
            )
            a_t, a_prev = alphas[index], alphas_prev[index]
            latent, pred_x0 = self.get_x_prev_and_pred_x0(
                latent, e_t, index, a_t, a_prev, temperature, seed
            )

        # Decoding stage
        decoded = self.decoder.predict_on_batch(latent)
        decoded = ((decoded + 1) / 2) * 255
        return np.clip(decoded, 0, 255).astype("uint8")

    def timestep_embedding(self, timesteps, dim=320, max_period=10000):
        half = dim // 2
        freqs = np.exp(
            -math.log(max_period) * np.arange(0, half, dtype="float32") / half
        )
        args = np.array(timesteps) * freqs
        embedding = np.concatenate([np.cos(args), np.sin(args)])
        return tf.convert_to_tensor(embedding.reshape(1, -1))

    def get_model_output(
        self,
        latent,
        t,
        context,
        unconditional_context,
        unconditional_guidance_scale,
        batch_size,
    ):
        timesteps = np.array([t])
        t_emb = self.timestep_embedding(timesteps)
        t_emb = np.repeat(t_emb, batch_size, axis=0)
        unconditional_latent = self.diffusion_model.predict_on_batch(
            [latent, t_emb, unconditional_context]
        )
        latent = self.diffusion_model.predict_on_batch([latent, t_emb, context])
        return unconditional_latent + unconditional_guidance_scale * (
            latent - unconditional_latent
        )

    def get_x_prev_and_pred_x0(self, x, e_t, index, a_t, a_prev, temperature, seed):
        sigma_t = 0
        sqrt_one_minus_at = math.sqrt(1 - a_t)
        pred_x0 = (x - sqrt_one_minus_at * e_t) / math.sqrt(a_t)

        # Direction pointing to x_t
        dir_xt = math.sqrt(1.0 - a_prev - sigma_t**2) * e_t
        noise = sigma_t * tf.random.normal(x.shape, seed=seed) * temperature
        x_prev = math.sqrt(a_prev) * pred_x0 + dir_xt
        return x_prev, pred_x0

    def get_starting_parameters(self, timesteps, batch_size, seed):
        n_h = self.img_height // 8
        n_w = self.img_width // 8
        alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
        alphas_prev = [1.0] + alphas[:-1]
        latent = tf.random.normal((batch_size, n_h, n_w, 4), seed=seed)
        return latent, alphas, alphas_prev


def get_models(img_height, img_width, max_text_length=77):
    # Create text encoder
    text_encoder = TextEncoder(max_text_length)

    # Creation diffusion UNet
    diffusion_model = DiffusionModel(img_height, img_width, max_text_length)

    # Create decoder
    decoder = Decoder(img_height, img_width)

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

    text_encoder.load_weights(text_encoder_weights_fpath)
    diffusion_model.load_weights(diffusion_model_weights_fpath)
    decoder.load_weights(decoder_weights_fpath)
    return text_encoder, diffusion_model, decoder


if __name__ == "__main__":
    # TODO: delete later, this is for testing only
    stablediff = StableDiffusion()
    img = stablediff.text_to_image("an astronaut riding a horse")
    from PIL import Image

    Image.fromarray(img[0]).save("test.png")
    print(f"saved at test.png")
