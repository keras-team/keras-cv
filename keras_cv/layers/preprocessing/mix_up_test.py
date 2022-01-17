import tensorflow as tf
from keras_cv.layers.preprocessing.mix_up import MixUp


NUM_CLASSES = 10


class MixUpTest(tf.test.TestCase):
    def test_return_shapes(self):
        xs = tf.ones((2, 512, 512, 3))
        # randomly sample labels
        ys = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 2)
        ys = tf.squeeze(ys)
        ys = tf.one_hot(ys, NUM_CLASSES)

        layer = MixUp(1.0)
        xs, ys = layer(xs, ys)

        self.assertEqual(xs.shape, [2, 512, 512, 3])
        # one hot smoothed labels
        self.assertEqual(ys.shape, [2, 10])
        self.assertEqual(len(ys != 0.0), 2)

    def test_label_smoothing(self):
        xs = tf.ones((2, 512, 512, 3))
        # randomly sample labels
        ys = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 2)
        ys = tf.squeeze(ys)
        ys = tf.one_hot(ys, NUM_CLASSES)

        layer = MixUp(1.0, label_smoothing=0.2)
        xs, ys = layer(xs, ys)
        self.assertNotAllClose(ys, 0.0)

    def test_mix_up_call_results(self):
        xs = tf.cast(
            tf.stack([2 * tf.ones((4, 4, 3)), tf.ones((4, 4, 3))], axis=0,), tf.float32,
        )
        ys = tf.one_hot(tf.constant([0, 1]), 2)

        layer = MixUp(1.0, label_smoothing=0.0)
        xs, ys = layer(xs, ys)

        # None of the individual values should still be close to 1 or 0
        self.assertNotAllClose(xs, 1.0)
        self.assertNotAllClose(xs, 2.0)

        # No labels should still be close to their originals
        self.assertNotAllClose(ys, 1.0)
        self.assertNotAllClose(ys, 0.0)

    def test_in_tf_function(self):
        xs = tf.cast(
            tf.stack([2 * tf.ones((4, 4, 3)), tf.ones((4, 4, 3))], axis=0,), tf.float32,
        )
        ys = tf.one_hot(tf.constant([0, 1]), 2)

        layer = MixUp(1.0, label_smoothing=0.0)

        @tf.function
        def augment(x, y):
            return layer(x, y)
        xs, ys = augment(xs, ys)

        # None of the individual values should still be close to 1 or 0
        self.assertNotAllClose(xs, 1.0)
        self.assertNotAllClose(xs, 2.0)

        # No labels should still be close to their originals
        self.assertNotAllClose(ys, 1.0)
        self.assertNotAllClose(ys, 0.0)
