import tensorflow as tf
from keras_cv.layers.preprocessing.random_mix_up import RandomMixUp

class RandomMixUpTest(tf.test.TestCase):
    def test_return_shapes(self):
        xs = tf.ones((2, 512, 512, 3))
        # randomly sample labels
        ys = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 2)
        ys = tf.squeeze(ys, axis=0)

        layer = RandomMixUp(num_classes=2, probability=1.0) 
        xs, ys = layer((xs, ys))
        
        self.assertEqual(xs.shape, [2, 512, 512, 3])
        # one hot smoothed labels
        self.assertEqual(ys.shape, [2, 2])
