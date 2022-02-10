
import tensorflow as tf 
from tensorflow.keras import layers, backend

class ChannelShuffle(layers.Layer):
    def __init__(
        self,
        seed=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.seed = seed

    @tf.function
    def _channel_shuffling(self, image):
        x = tf.transpose(image)
        x = tf.random.shuffle(x, seed=self.seed)
        return tf.transpose(x)

    def call(self, images, training=None):
        if training is None:
            training = backend.learning_phase()

        if not training:
            return images
  
        unbatched = images.shape.rank == 3
        if unbatched:
            images = tf.expand_dims(images, axis=0)

        # TODO: Make the batch operation vectorize.
        output = tf.map_fn(lambda image: self._channel_shuffling(image), images)

        if unbatched:
            output = tf.squeeze(output, axis=0)
        return output