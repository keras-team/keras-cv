#!/bin/bash

SETUP="
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

import keras_cv.point_cloud

num_points = 200000
num_boxes = 1000
box_dimension = 20.0

def get_points_boxes():
    points = tf.random.uniform(
        shape=[num_points, 2], minval=0, maxval=box_dimension, dtype=tf.float32
    )
    points_z = 5.0 * tf.ones(shape=[num_points, 1], dtype=tf.float32)
    points = tf.concat([points, points_z], axis=-1)
    boxes_x = tf.random.uniform(
        shape=[num_boxes, 1],
        minval=0,
        maxval=box_dimension - 1.0,
        dtype=tf.float32,
    )
    boxes_y = tf.random.uniform(
        shape=[num_boxes, 1],
        minval=0,
        maxval=box_dimension - 1.0,
        dtype=tf.float32,
    )
    boxes_dx = tf.random.uniform(
        shape=[num_boxes, 1], minval=0, maxval=5.0, dtype=tf.float32
    )
    boxes_dx = tf.math.minimum(box_dimension - boxes_x, boxes_dx)
    boxes_dy = tf.random.uniform(
        shape=[num_boxes, 1], minval=0, maxval=5.0, dtype=tf.float32
    )
    boxes_dy = tf.math.minimum(box_dimension - boxes_y, boxes_dy)
    boxes_z = 5.0 * tf.ones([num_boxes, 1], dtype=tf.float32)
    boxes_dz = 3.0 * tf.ones([num_boxes, 1], dtype=tf.float32)
    boxes_angle = tf.zeros([num_boxes, 1], dtype=tf.float32)
    boxes = tf.concat(
        [boxes_x, boxes_y, boxes_z, boxes_dx, boxes_dy, boxes_dz, boxes_angle],
        axis=-1,
    )
    return points, boxes

points, boxes = get_points_boxes();
"

echo "----------------------------------------"
echo "benchmark_within_any_box3d"
python -m timeit -s "$SETUP" \
  "keras_cv.point_cloud.is_within_any_box3d(points, boxes)"

echo "----------------------------------------"
echo benchmark_within_any_box3d_v2
python -m timeit -s "$SETUP" \
  "keras_cv.point_cloud.is_within_any_box3d_v2(points, boxes)"

echo "----------------------------------------"
echo benchmark_within_any_box3d_v3
python -m timeit -s "$SETUP" \
  "keras_cv.point_cloud.is_within_any_box3d_v3(points, boxes)"


