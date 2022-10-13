import tensorflow as tf
from tensorflow.keras import layers


# Referred from: github.com:rwightman/pytorch-image-models.
# Referred from: https://github.com/sayakpaul/swin-transformers-tf
class StochasticDepth(layers.Layer):
    def __init__(self, drop_prop, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = float(drop_prop)

    def call(self, x, training=False):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],)+(1,)*(tf.shape(x).shape[0]-1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_prob": self.drop_prob})
        return config