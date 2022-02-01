import tensorflow as tf

from keras_cv.utils import fill_utils


class FillRectangleTest(tf.test.TestCase):
    def test_rectangle_position(self):
        batch_size = 1
        img_h, img_w = 8, 8
        rec_h, rec_w = 3, 5
        cent_x, cent_y = 2, 3

        batch_shape = (batch_size, img_h, img_w, 1)
        images = tf.ones(batch_shape)

        centers_x = tf.fill([batch_size], cent_x)
        centers_y = tf.fill([batch_size], cent_y)
        height = tf.fill([batch_size], rec_h)
        width = tf.fill([batch_size], rec_w)
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

    def test_width_out_of_lower_bound(self):
        batch_size = 1
        img_h, img_w = 8, 8
        rec_h, rec_w = 3, 5
        cent_x, cent_y = 1, 3

        batch_shape = (batch_size, img_h, img_w, 1)
        images = tf.ones(batch_shape)

        centers_x = tf.fill([batch_size], cent_x)
        centers_y = tf.fill([batch_size], cent_y)
        height = tf.fill([batch_size], rec_h)
        width = tf.fill([batch_size], rec_w)
        fill = tf.zeros_like(images)

        filled_images = fill_utils.fill_rectangle(
            images, centers_x, centers_y, width, height, fill
        )
        # remove batch dimension and channel dimension
        filled_images = filled_images[0, ..., 0]

        # assert width is truncated (from 5 to 4) when cent_x - rec_w < 0 (1 - 5 < 0)
        expected = tf.constant(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=images.dtype,
        )
        tf.assert_equal(filled_images, expected)

    def test_width_out_of_upper_bound(self):
        batch_size = 1
        img_h, img_w = 8, 8
        rec_h, rec_w = 3, 5
        cent_x, cent_y = 6, 3

        batch_shape = (batch_size, img_h, img_w, 1)
        images = tf.ones(batch_shape)

        centers_x = tf.fill([batch_size], cent_x)
        centers_y = tf.fill([batch_size], cent_y)
        height = tf.fill([batch_size], rec_h)
        width = tf.fill([batch_size], rec_w)
        fill = tf.zeros_like(images)

        filled_images = fill_utils.fill_rectangle(
            images, centers_x, centers_y, width, height, fill
        )
        # remove batch dimension and channel dimension
        filled_images = filled_images[0, ..., 0]

        # assert width is truncated (from 5 to 4) when cent_x + rec_w > img_w (6 + 5 > 8)
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
