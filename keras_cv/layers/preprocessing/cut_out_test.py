import tensorflow as tf

from keras_cv.layers.preprocessing import cut_out

NUM_CLASSES = 10


class CutOutTest(tf.test.TestCase):
    def test_return_shapes(self):
        xs = tf.ones((2, 512, 512, 3))

        layer = cut_out.CutOut(1.0, length=50, fill_value=0.0, seed=1)
        xs = layer(xs)

        self.assertEqual(xs.shape, [2, 512, 512, 3])

    def test_cut_out_call_results(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((4, 4, 3)), tf.ones((4, 4, 3))],
                axis=0,
            ),
            tf.float32,
        )

        patch_value = 0.0
        layer = cut_out.CutOut(1.0, length=2, fill_value=patch_value, seed=1)
        xs = layer(xs)

        # At least some pixels should be replaced in the CutOut operation
        self.assertTrue(tf.math.reduce_any(xs[0] == patch_value))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == patch_value))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))

    def test_cut_out_call_results_one_channel(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((4, 4, 1)), tf.ones((4, 4, 1))],
                axis=0,
            ),
            tf.float32,
        )

        patch_value = 0.0
        layer = cut_out.CutOut(1.0, length=2, fill_value=patch_value, seed=1)
        xs = layer(xs)

        # At least some pixels should be replaced in the CutOut operation
        self.assertTrue(tf.math.reduce_any(xs[0] == patch_value))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == patch_value))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))

    def test_in_tf_function(self):
        xs = tf.cast(
            tf.stack([2 * tf.ones((100, 100, 1)), tf.ones((100, 100, 1))], axis=0),
            tf.float32,
        )

        patch_value = 0.0
        layer = cut_out.CutOut(1.0, length=50, fill_value=patch_value, seed=1)

        @tf.function
        def augment(x):
            return layer(x)

        xs = augment(xs)

        # At least some pixels should be replaced in the CutOut operation
        self.assertTrue(tf.math.reduce_any(xs[0] == patch_value))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == patch_value))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))
