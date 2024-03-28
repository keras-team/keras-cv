import keras.saving
import numpy as np
import pytest
import os

from keras_cv.models.feature_extractor.coca import CoCa
from keras_cv.tests.test_case import TestCase

class CoCaTest(TestCase):

    @pytest.mark.large
    def test_coca_model_save(self):
        # TODO: Transformer encoder breaks if you have project dim < num heads
        model = CoCa()

        save_path = os.path.join(self.get_temp_dir(), "coca.keras")
        model.save(save_path)

        restored_model = keras.models.load_model(save_path, custom_objects={"CoCa": CoCa})

        self.assertIsInstance(restored_model, CoCa)


