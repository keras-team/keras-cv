import tensorflow as tf
from absl.testing import parameterized


class ResizingTest(tf.test.TestCase, parameterized.TestCase):
    # TODO(lukewood): figure out the set of cases we need to port from core Keras.
    def test_resize_with_distortion_with_boxes(self):
        pass

    def test_resize_with_pad_with_boxes(self):
        pass

    def test_resize_with_crop_rejects_boxes(self):
        pass
