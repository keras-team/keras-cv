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

import os

import numpy as np
import pytest

from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.models.backbones.video_swin.video_swin_backbone import (
    VideoSwinBackbone,
)
from keras_cv.tests.test_case import TestCase


class TestVideoSwinSBackbone(TestCase):

    @pytest.mark.large
    def test_call(self):
        model = VideoSwinBackbone(  # TODO: replace with aliases
            include_rescaling=True, input_shape=(8, 256, 256, 3)
        )
        x = np.ones((1, 8, 256, 256, 3))
        x_out = ops.convert_to_numpy(model(x))
        num_parameters = sum(
            np.prod(tuple(x.shape)) for x in model.trainable_variables
        )
        self.assertEqual(x_out.shape, (1, 4, 8, 8, 768))
        self.assertEqual(num_parameters, 27_663_894)

    @pytest.mark.extra_large
    def teat_save(self):
        # saving test
        model = VideoSwinBackbone(include_rescaling=False)
        x = np.ones((1, 32, 224, 224, 3))
        x_out = ops.convert_to_numpy(model(x))
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path)
        loaded_model = keras.saving.load_model(path)
        x_out_loaded = ops.convert_to_numpy(loaded_model(x))
        self.assertAllClose(x_out, x_out_loaded)

    @pytest.mark.extra_large
    def test_fit(self):
        model = VideoSwinBackbone(include_rescaling=False)
        x = np.ones((1, 32, 224, 224, 3))
        y = np.zeros((1, 16, 7, 7, 768))
        model.compile(optimizer="adam", loss="mse", metrics=["mse"])
        model.fit(x, y, epochs=1)

    @pytest.mark.extra_large
    def test_can_run_in_mixed_precision(self):
        keras.mixed_precision.set_global_policy("mixed_float16")
        model = VideoSwinBackbone(
            include_rescaling=False, input_shape=(8, 224, 224, 3)
        )
        x = np.ones((1, 8, 224, 224, 3))
        y = np.zeros((1, 4, 7, 7, 768))
        model.compile(optimizer="adam", loss="mse", metrics=["mse"])
        model.fit(x, y, epochs=1)

    @pytest.mark.extra_large
    def test_can_run_on_gray_video(self):
        model = VideoSwinBackbone(
            include_rescaling=False,
            input_shape=(96, 96, 96, 1),
            window_size=[6, 6, 6],
        )
        x = np.ones((1, 96, 96, 96, 1))
        y = np.zeros((1, 48, 3, 3, 768))
        model.compile(optimizer="adam", loss="mse", metrics=["mse"])
        model.fit(x, y, epochs=1)
