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

import numpy as np

from keras_cv.src.backend import ops
from keras_cv.src.models.stable_diffusion_v3.model_sampling import ModelSampling
from keras_cv.src.tests.test_case import TestCase


class ModelSamplingTest(TestCase):
    def test_sigma_stats(self):
        model_sampling = ModelSampling()
        self.assertEqual(model_sampling.sigma_min, 0.001)
        self.assertEqual(model_sampling.sigma_max, 1.0)

    def test_timestep(self):
        steps = 50
        model_sampling = ModelSampling()
        start = model_sampling.timestep(model_sampling.sigma_max)
        end = model_sampling.timestep(model_sampling.sigma_min)
        timesteps = ops.linspace(start, end, steps)

        golden_start = [1000.0, 979.61224, 959.2245, 938.83673, 918.449]
        golden_end = [82.55102, 62.16326, 41.77551, 21.387754, 1.0]
        self.assertAllClose(timesteps[:5], golden_start)
        self.assertAllClose(timesteps[-5:], golden_end, atol=1e-4)

    def test_sigma(self):
        timesteps = np.linspace(1000, 1, 50)
        model_sampling = ModelSampling()
        sigmas = model_sampling.sigma(timesteps)
        sigmas = ops.pad(sigmas, [0, 1])

        golden_start = [1.0, 0.97961223, 0.95922446, 0.93883675, 0.918449]
        golden_end = [0.06216326, 0.04177551, 0.02138775, 0.001, 0.0]
        self.assertAllClose(sigmas[:5], golden_start)
        self.assertAllClose(sigmas[-5:], golden_end)

    def test_noise_scaling(self):
        latent = np.ones((1, 40, 40, 16)).astype("float32") * 0.0609
        noise = np.zeros((1, 40, 40, 16)).astype("float32")  # Fixed noise
        sigma = 0.5
        model_sampling = ModelSampling()
        noise_scaled = model_sampling.noise_scaling(sigma, noise, latent)
        self.assertAllClose(
            noise_scaled, sigma * noise + (1.0 - sigma) * latent
        )
