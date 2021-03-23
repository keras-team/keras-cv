"""Tests for iou_similarity.py."""

import tensorflow as tf

from keras_cv.ops import iou_similarity


class BoxMatcherTest(tf.test.TestCase):

  def test_similarity_unbatched(self):
    boxes = tf.constant(
        [
            [0, 0, 1, 1],
            [5, 0, 10, 5],
        ],
        dtype=tf.float32)

    gt_boxes = tf.constant(
        [
            [0, 0, 5, 5],
            [0, 5, 5, 10],
            [5, 0, 10, 5],
            [5, 5, 10, 10],
        ],
        dtype=tf.float32)

    sim_calc = iou_similarity.IouSimilarity()
    sim_matrix = sim_calc(boxes, gt_boxes)

    self.assertAllClose(
        sim_matrix.numpy(),
        [[0.04, 0, 0, 0],
         [0, 0, 1., 0]])

  def test_similarity_batched(self):
    boxes = tf.constant(
        [[
            [0, 0, 1, 1],
            [5, 0, 10, 5],
        ]],
        dtype=tf.float32)

    gt_boxes = tf.constant(
        [[
            [0, 0, 5, 5],
            [0, 5, 5, 10],
            [5, 0, 10, 5],
            [5, 5, 10, 10],
        ]],
        dtype=tf.float32)

    sim_calc = iou_similarity.IouSimilarity()
    sim_matrix = sim_calc(boxes, gt_boxes)

    self.assertAllClose(
        sim_matrix.numpy(),
        [[[0.04, 0, 0, 0],
          [0, 0, 1., 0]]])


if __name__ == '__main__':
  tf.test.main()
