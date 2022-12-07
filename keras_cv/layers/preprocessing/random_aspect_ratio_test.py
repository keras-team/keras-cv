import tensorflow as tf

from keras_cv import layers


class RandomAspectRatioTest(tf.test.TestCase):
    def test_train_augments_image(self):
        # Checks if original and augmented images are different
        input_image_shape = (8, 100, 100, 3)
        image = tf.random.uniform(shape=input_image_shape)

        layer = layers.RandomAspectRatio(factor=(0.9, 1.1))
        output = layer(image, training=True)
        self.assertNotEqual(output.shape, input_image_resized.shape)

    def test_inference_preserves_image(self):
        # Checks if original and augmented images are different
        input_image_shape = (8, 100, 100, 3)
        image = tf.random.uniform(shape=input_image_shape)

        layer = layers.RandomAspectRatio(factor=(0.9, 1.1))
        output = layer(image, training=False)
        self.assertAllClose(input, output)

    def test_grayscale(self):
        # Checks if original and augmented images are different
        input_image_shape = (8, 100, 100, 1)
        image = tf.random.uniform(shape=input_image_shape, seed=1223)

        layer = layers.RandomAspectRatio(factor=(0.9, 1.1))
        output = layer(image, training=True)
        self.assertNotEqual(output.shape[-1], 1)

    def test_augment_boxes_ragged(self):
        image = tf.zeros([2, 20, 20, 3])
        boxes = tf.ragged.constant(
            [[[0, 0, 1, 1], [0, 0, 1, 1]], [[0, 0, 1, 1]]], dtype=tf.float32
        )
        boxes = bounding_box.add_class_id(boxes)
        input = {"images": image, "bounding_boxes": boxes}
        layer = layers.RandomAspectRatio(
            factor=(0.9, 1.1), bounding_box_format="rel_xywh"
        )
        output = layer(input, training=True)

        # the result boxes will still have the entire image in them
        expected_output = tf.ragged.constant(
            [[[0, 0, 1, 1, 0], [0, 0, 1, 1, 0]], [[0, 0, 1, 1, 0]]], dtype=tf.float32
        )
        self.assertAllClose(
            expected_output.to_tensor(-1), output["bounding_boxes"].to_tensor(-1)
        )
