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
"""StableDiffusion Noise scheduler

Adapted from https://github.com/huggingface/diffusers/blob/v0.3.0/src/diffusers/schedulers/scheduling_ddpm.py#L56
"""  # noqa: E501

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import ops
from keras_cv.src.backend import random


@keras_cv_export("keras_cv.models.stable_diffusion.NoiseScheduler")
class NoiseScheduler:
    """
    Args:
        train_timesteps: number of diffusion steps used to train the model.
        beta_start: the starting `beta` value of inference.
        beta_end: the final `beta` value.
        beta_schedule: the beta schedule, a mapping from a beta range to a
            sequence of betas for stepping the model. Choose from `linear` or
            `quadratic`.
        variance_type: options to clip the variance used when adding noise to
            the de-noised sample. Choose from `fixed_small`, `fixed_small_log`,
            `fixed_large`, `fixed_large_log`, `learned` or `learned_range`.
        clip_sample: option to clip predicted sample between -1 and 1 for
            numerical stability.
    """

    def __init__(
        self,
        train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        variance_type="fixed_small",
        clip_sample=True,
    ):
        self.train_timesteps = train_timesteps

        if beta_schedule == "linear":
            self.betas = ops.linspace(beta_start, beta_end, train_timesteps)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                ops.linspace(beta_start**0.5, beta_end**0.5, train_timesteps)
                ** 2
            )
        else:
            raise ValueError(f"Invalid beta schedule: {beta_schedule}.")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = ops.cumprod(self.alphas)

        self.variance_type = variance_type
        self.clip_sample = clip_sample
        self.seed_generator = random.SeedGenerator(seed=42)

    def _get_variance(self, timestep, predicted_variance=None):
        alpha_prod = self.alphas_cumprod[timestep]
        alpha_prod_prev = (
            self.alphas_cumprod[timestep - 1] if timestep > 0 else 1.0
        )

        variance = (
            (1 - alpha_prod_prev) / (1 - alpha_prod) * self.betas[timestep]
        )

        if self.variance_type == "fixed_small":
            variance = ops.clip(variance, x_min=1e-20, x_max=1)
        elif self.variance_type == "fixed_small_log":
            variance = ops.log(ops.clip(variance, x_min=1e-20, x_max=1))
        elif self.variance_type == "fixed_large":
            variance = self.betas[timestep]
        elif self.variance_type == "fixed_large_log":
            variance = ops.log(self.betas[timestep])
        elif self.variance_type == "learned":
            return predicted_variance
        elif self.variance_type == "learned_range":
            min_log = variance
            max_log = self.betas[timestep]
            frac = (predicted_variance + 1) / 2
            variance = frac * max_log + (1 - frac) * min_log
        else:
            raise ValueError(f"Invalid variance type: {self.variance_type}")

        return variance

    def step(
        self,
        model_output,
        timestep,
        sample,
        predict_epsilon=True,
    ):
        """
        Predict the sample at the previous timestep by reversing the SDE. Core
        function to propagate the diffusion process from the learned model
        outputs (usually the predicted noise).
        Args:
            model_output: a Tensor containing direct output from learned
                diffusion model
            timestep: current discrete timestep in the diffusion chain.
            sample: a Tensor containing the current instance of sample being
                created by diffusion process.
            predict_epsilon: whether the model is predicting noise (epsilon) or
                samples
        Returns:
            The predicted sample at the previous timestep
        """

        if model_output.shape[1] == sample.shape[
            1
        ] * 2 and self.variance_type in [
            "learned",
            "learned_range",
        ]:
            model_output, predicted_variance = ops.split(
                model_output, sample.shape[1], axis=1
            )
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod = self.alphas_cumprod[timestep]
        alpha_prod_prev = (
            self.alphas_cumprod[timestep - 1] if timestep > 0 else 1.0
        )
        beta_prod = 1 - alpha_prod
        beta_prod_prev = 1 - alpha_prod_prev

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf  # noqa: E501
        if predict_epsilon:
            pred_original_sample = (
                sample - beta_prod ** (0.5) * model_output
            ) / alpha_prod ** (0.5)
        else:
            pred_original_sample = model_output

        # 3. Clip "predicted x_0"
        if self.clip_sample:
            pred_original_sample = ops.clip_by_value(
                pred_original_sample, -1, 1
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current
        # sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (
            alpha_prod_prev ** (0.5) * self.betas[timestep]
        ) / beta_prod
        current_sample_coeff = (
            self.alphas[timestep] ** (0.5) * beta_prod_prev / beta_prod
        )

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * sample
        )

        # 6. Add noise
        variance = 0
        if timestep > 0:
            noise = random.normal(model_output.shape, seed=self.seed_generator)
            variance = (
                self._get_variance(
                    timestep, predicted_variance=predicted_variance
                )
                ** 0.5
            ) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(
        self,
        original_samples,
        noise,
        timesteps,
    ):
        sqrt_alpha_prod = ops.take(self.alphas_cumprod, timesteps) ** 0.5
        sqrt_one_minus_alpha_prod = (
            1 - ops.take(self.alphas_cumprod, timesteps)
        ) ** 0.5

        for _ in range(3):
            sqrt_alpha_prod = ops.expand_dims(sqrt_alpha_prod, axis=-1)
            sqrt_one_minus_alpha_prod = ops.expand_dims(
                sqrt_one_minus_alpha_prod, axis=-1
            )
        sqrt_alpha_prod = ops.cast(
            sqrt_alpha_prod, dtype=original_samples.dtype
        )
        sqrt_one_minus_alpha_prod = ops.cast(
            sqrt_one_minus_alpha_prod, dtype=noise.dtype
        )
        noisy_samples = (
            sqrt_alpha_prod * original_samples
            + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples

    def __len__(self):
        return self.train_timesteps
