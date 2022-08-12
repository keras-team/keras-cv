
import tensorflow as tf

from keras_cv.layers import NonMaxSuppression
from absl.testing import parameterized

class AnchorGeneratorTest(tf.test.TestCase, parameterized.TestCase):

    @parameterized.named_parameters(
        ([0, 1, 2], [1]),
        ({'level_1': [0, 1, 2]}, {'1': [0, 1, 2]}),
    )
    def test_raises_when_strides_not_equal_to_sizes(self, anchor_sizes, strides):
        # construct generator, check assertion
        pass

    def test_output_shapes(self):
        pass

    def test_bounding_box_formats(self):
        pass
