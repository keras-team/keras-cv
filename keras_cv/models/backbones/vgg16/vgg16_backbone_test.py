import os

import numpy as np
import pytest

from keras_cv.backend import keras
from keras_cv.models import VGG16Backbone
from keras_cv.tests.test_case import TestCase


class VGG16BackboneTest(TestCase):
    def setUp(self):
        self.img_input = np.ones((2, 224, 224, 3), dtype="float32")

    def test_valid_call(self):
        model = VGG16Backbone(
            input_shape=(224, 224, 3),
            include_top=False,
            include_rescaling=False,
            pooling="avg",
        )
        model(self.img_input)

    def test_valid_call_with_rescaling(self):
        model = VGG16Backbone(
            input_shape=(224, 224, 3),
            include_top=False,
            include_rescaling=True,
            pooling="avg",
        )
        model(self.img_input)

    def test_valid_call_with_top(self):
        model = VGG16Backbone(
            input_shape=(224, 224, 3),
            include_top=True,
            include_rescaling=False,
            num_classes=2,
        )
        model(self.img_input)

    @pytest.mark.large
    def test_saved_model(self):
        model = VGG16Backbone(
            input_shape=(224, 224, 3),
            include_top=False,
            include_rescaling=False,
            num_classes=2,
            pooling="avg",
        )
        model_output = model(self.img_input)
        save_path = os.path.join(
            self.get_temp_dir(), "vgg16.keras"
        )
        model.save(save_path)
        restored_model = keras.models.load_model(save_path)

        # Check the restored model is instance of VGG16Backbone
        self.assertIsInstance(restored_model, VGG16Backbone)

        # Check if the restored model gives the same output
        restored_model_output = restored_model(self.img_input)
        self.assertAllClose(model_output, restored_model_output)
