import tensorflow as tf
from absl.testing import parameterized

from keras_cv.models import regnet

from .models_test import ModelsTest

MODEL_LIST = [
    (regnet.RegNetX002, 368, {}),
    (regnet.RegNetX004, 384, {}),
    (regnet.RegNetX006, 528, {}),
    (regnet.RegNetX008, 672, {}),
    (regnet.RegNetX016, 912, {}),
    (regnet.RegNetX032, 1008, {}),
    (regnet.RegNetX040, 1360, {}),
    (regnet.RegNetX064, 1624, {}),
    (regnet.RegNetX080, 1920, {}),
    (regnet.RegNetX120, 2240, {}),
    (regnet.RegNetX160, 2048, {}),
    (regnet.RegNetX320, 2520, {}),
]


class ResNetV2Test(ModelsTest, tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters(*MODEL_LIST)
    def test_application_base(self, app, _, args):
        super()._test_application_base(app, _, args)

    @parameterized.parameters(*MODEL_LIST)
    def test_application_with_rescaling(self, app, last_dim, args):
        super()._test_application_with_rescaling(app, last_dim, args)

    @parameterized.parameters(*MODEL_LIST)
    def test_application_pooling(self, app, last_dim, args):
        super()._test_application_pooling(app, last_dim, args)

    @parameterized.parameters(*MODEL_LIST)
    def test_application_variable_input_channels(self, app, last_dim, args):
        super()._test_application_variable_input_channels(app, last_dim, args)

    @parameterized.parameters(*MODEL_LIST)
    def test_model_can_be_used_as_backbone(self, app, last_dim, args):
        super()._test_model_can_be_used_as_backbone(app, last_dim, args)


if __name__ == "__main__":
    tf.test.main()
