import tensorflow as tf

from keras_cv.utils import fill_utils


class FillRectangleTest(tf.test.TestCase):
    def test_rectangle_position(self):
        batch_size = 1
        h, w = 8, 8
        rh, rw = 3, 5
        cx, cy = 2, 3

        batch_shape = (batch_size, h, w, 1)
        images = tf.ones(batch_shape)

        centers_x = tf.fill([batch_size], cx)
        centers_y = tf.fill([batch_size], cy)
        height = tf.fill([batch_size], rh)
        width = tf.fill([batch_size], rw)
        fill = tf.zeros_like(images)

        filled_images = fill_utils.fill_rectangle(
            images, centers_x, centers_y, width, height, fill
        )
        # remove batch dimension and channel dimension
        filled_images = filled_images[0, ..., 0]

        expected = tf.constant(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=images.dtype,
        )
        tf.assert_equal(filled_images, expected)


    def test_center_plus_width_above_bounds(self):
        batch_size = 1
        h, w = 8, 8
        rh, rw = 3, 5
        cx, cy = 6, 3

        batch_shape = (batch_size, h, w, 1)
        images = tf.ones(batch_shape)

        centers_x = tf.fill([batch_size], cx)
        centers_y = tf.fill([batch_size], cy)
        height = tf.fill([batch_size], rh)
        width = tf.fill([batch_size], rw)
        fill = tf.zeros_like(images)

        filled_images = fill_utils.fill_rectangle(
            images, centers_x, centers_y, width, height, fill
        )
        # remove batch dimension and channel dimension
        filled_images = filled_images[0, ..., 0]

        expected = tf.constant(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=images.dtype,
        )
        tf.assert_equal(filled_images, expected)

    def test_center_minus_width_below_bounds(self):
        batch_size = 1
        h, w = 8, 8
        rh, rw = 3, 5  # TODO
        cx, cy = 6, 3  # TODO

        batch_shape = (batch_size, h, w, 1)
        images = tf.ones(batch_shape)

        centers_x = tf.fill([batch_size], cx)
        centers_y = tf.fill([batch_size], cy)
        height = tf.fill([batch_size], rh)
        width = tf.fill([batch_size], rw)
        fill = tf.zeros_like(images)

        filled_images = fill_utils.fill_rectangle(
            images, centers_x, centers_y, width, height, fill
        )
        # remove batch dimension and channel dimension
        filled_images = filled_images[0, ..., 0]

        # TODO:
        expected = tf.constant(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=images.dtype,
        )
        tf.assert_equal(filled_images, expected)
