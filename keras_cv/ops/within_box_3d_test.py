import os
from datetime import datetime

import tensorflow as tf

import keras_cv

# os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

num_points = 200000
num_boxes = 1000
box_dimension = 20.0


@tf.function
def is_within_box2d(points, boxes):
    return keras_cv.ops.is_within_box3d_v2(points, boxes)


def get_points_boxes():
    points = tf.random.uniform(
        shape=[num_points, 2], minval=0, maxval=box_dimension, dtype=tf.float32
    )
    points_z = 5.0 * tf.ones(shape=[num_points, 1], dtype=tf.float32)
    points = tf.concat([points, points_z], axis=-1)
    boxes_x = tf.random.uniform(
        shape=[num_boxes, 1], minval=0, maxval=box_dimension - 1.0, dtype=tf.float32
    )
    boxes_y = tf.random.uniform(
        shape=[num_boxes, 1], minval=0, maxval=box_dimension - 1.0, dtype=tf.float32
    )
    boxes_dx = tf.random.uniform(
        shape=[num_boxes, 1], minval=0, maxval=5.0, dtype=tf.float32
    )
    boxes_dx = tf.math.minimum(10 - boxes_x, boxes_dx)
    boxes_dy = tf.random.uniform(
        shape=[num_boxes, 1], minval=0, maxval=5.0, dtype=tf.float32
    )
    boxes_dy = tf.math.minimum(10 - boxes_y, boxes_dy)
    boxes_z = 5.0 * tf.ones([num_boxes, 1], dtype=tf.float32)
    boxes_dz = 3.0 * tf.ones([num_boxes, 1], dtype=tf.float32)
    boxes_angle = tf.zeros([num_boxes, 1], dtype=tf.float32)
    boxes = tf.concat(
        [boxes_x, boxes_y, boxes_z, boxes_dx, boxes_dy, boxes_dz, boxes_angle], axis=-1
    )
    return points, boxes


with tf.device("cpu:0"):

    points, boxes = get_points_boxes()

    for i in range(2):
        print(datetime.now())
        res = is_within_box2d(points, boxes)
        print(tf.unique(res)[0])
        # if i == 0:
        #     tf.profiler.experimental.start('./logs/withinbox_v1')
#        res = tf.cast(res, tf.float32)
#        print(tf.reduce_max(res))
#        print(tf.reduce_min(res))
# tf.profiler.experimental.stop()
