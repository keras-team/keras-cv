# Copyright 2024 The KerasCV Authors
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

from keras_cv.src.backend import ops


class ModelSampling:
    """Helper for sampler scheduling for Discrete Flow models"""

    def __init__(self, shift=1.0):
        self.shift = shift
        self.timesteps = 1000
        self.sigmas = self.sigma(
            ops.arange(1, self.timesteps + 1, 1, dtype="float32")
        )
        self.sigma_min = self.sigmas[0]
        self.sigma_max = self.sigmas[-1]

    def timestep(self, sigma):
        return ops.multiply(sigma, self.timesteps)

    def sigma(self, timestep):
        timestep = ops.divide(timestep, self.timesteps)

        if self.shift == 1.0:
            sigmas = timestep
        else:
            sigmas = self.shift * timestep / (1 + (self.shift - 1) * timestep)
        return sigmas

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = ops.convert_to_tensor(sigma)
        model_output = ops.convert_to_tensor(model_output)
        model_input = ops.convert_to_tensor(model_input)

        ndim = model_output.ndim
        shape = (-1) + (1,) * (ndim - 1)
        sigma = ops.reshape(sigma, shape)
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image):
        sigma = ops.convert_to_tensor(sigma)
        noise = ops.convert_to_tensor(noise)
        latent_image = ops.convert_to_tensor(latent_image)
        return sigma * noise + (1.0 - sigma) * latent_image
