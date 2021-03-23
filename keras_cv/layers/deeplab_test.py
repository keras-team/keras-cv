"""Tests for ASPP."""

import tensorflow as tf

from tensorflow.python.keras import keras_parameterized
from keras_cv.layers import deeplab


@keras_parameterized.run_all_keras_modes
class DeeplabTest(keras_parameterized.TestCase):

  @keras_parameterized.parameterized.parameters(
      (None,),
      ([32, 32],),
      )
  def test_aspp(self, pool_kernel_size):
    inputs = tf.keras.Input(shape=(64, 64, 128), dtype=tf.float32)
    layer = deeplab.SpatialPyramidPooling(output_channels=256,
                                          dilation_rates=[6, 12, 18],
                                          pool_kernel_size=None)
    output = layer(inputs)
    self.assertAllEqual([None, 64, 64, 256], output.shape)

  def test_aspp_invalid_shape(self):
    inputs = tf.keras.Input(shape=(64, 64), dtype=tf.float32)
    layer = deeplab.SpatialPyramidPooling(output_channels=256,
                                          dilation_rates=[6, 12, 18])
    with self.assertRaises(ValueError):
      _ = layer(inputs)

  def test_config_with_custom_name(self):
    layer = deeplab.SpatialPyramidPooling(256, [5], name='aspp')
    config = layer.get_config()
    layer_1 = deeplab.SpatialPyramidPooling.from_config(config)
    self.assertEqual(layer_1.name, layer.name)


if __name__ == '__main__':
  tf.test.main()
