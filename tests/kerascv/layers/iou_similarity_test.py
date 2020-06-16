import numpy as np
import tensorflow as tf
from kerascv.layers.iou_similarity import IOUSimilarity


def test_iou_basic_absolute_coordinate():
    # both gt box and two anchors are size 4
    # the intersection between gt box and first anchor is 1 and union is 7
    # the intersection between gt box and second anchor is 0 and union is 8
    gt_boxes = tf.constant([[[0, 2, 2, 4]]])
    anchors = tf.constant([[1, 1, 3, 3], [-2, 1, 0, 3]])
    iou_layer = IOUSimilarity()
    similarity = iou_layer(gt_boxes, anchors)
    # batch_size = 1, n_gt_boxes = 1, n_anchors = 2
    expected_out = np.asarray([1 / 7, 0]).astype(np.float32).reshape((1, 1, 2))
    np.testing.assert_allclose(expected_out, similarity)


def test_iou_basic_normalized_coordinate():
    # both gt box and two anchors are size 1
    # the intersection between gt box and first anchor is 1 and union is 7
    # the intersection between gt box and second anchor is 0 and union is 8
    gt_boxes = tf.constant([[[0, 0.5, 0.5, 1.0]]])
    anchors = tf.constant([[0.25, 0.25, 0.75, 0.75], [-0.5, 0.25, 0, 0.75]])
    iou_layer = IOUSimilarity()
    similarity = iou_layer(gt_boxes, anchors)
    expected_out = np.asarray([1 / 7, 0]).astype(np.float32).reshape((1, 1, 2))
    np.testing.assert_allclose(expected_out, similarity)


def test_iou_multi_gt_multi_anchor_absolute_coordinate():
    # batch_size = 1, n_gt_boxes = 2
    # [1, 2, 4]
    gt_boxes = tf.constant([[[0, 2, 2, 4], [-1, 1, 1, 3]]])
    # [2, 4]
    anchors = tf.constant([[1, 1, 3, 3], [-2, 1, 0, 3]])
    iou_layer = IOUSimilarity()
    similarity = iou_layer(gt_boxes, anchors)
    expected_out = (
        np.asarray([[1 / 7, 0], [0, 1 / 3]]).astype(np.float32).reshape((1, 2, 2))
    )
    np.testing.assert_allclose(expected_out, similarity)


def test_iou_batched_gt_multi_anchor_absolute_coordinate():
    # batch_size = 2, n_gt_boxes = 1
    # [2, 1, 4]
    gt_boxes = tf.constant([[[0, 2, 2, 4]], [[-1, 1, 1, 3]]])
    # [2, 4]
    anchors = tf.constant([[1, 1, 3, 3], [-2, 1, 0, 3]])
    iou_layer = IOUSimilarity()
    similarity = iou_layer(gt_boxes, anchors)
    expected_out = (
        np.asarray([[1 / 7, 0], [0, 1 / 3]]).astype(np.float32).reshape((2, 1, 2))
    )
    np.testing.assert_allclose(expected_out, similarity)


def test_iou_batched_gt_batched_anchor_absolute_coordinate():
    # batch_size = 2, n_gt_boxes = 1
    # [2, 1, 4]
    gt_boxes = tf.constant([[[0, 2, 2, 4]], [[-1, 1, 1, 3]]])
    # [2, 1, 4]
    anchors = tf.constant([[[1, 1, 3, 3]], [[-2, 1, 0, 3]]])
    iou_layer = IOUSimilarity()
    similarity = iou_layer(gt_boxes, anchors)
    expected_out = np.asarray([[1 / 7], [1 / 3]]).astype(np.float32).reshape((2, 1, 1))
    np.testing.assert_allclose(expected_out, similarity)


def test_iou_multi_gt_multi_anchor_normalized_coordinate():
    # batch_size = 1, n_gt_boxes = 2
    # [1, 2, 4]
    gt_boxes = tf.constant([[[0.0, 0.5, 0.5, 1.0], [-0.25, 0.25, 0.25, 0.75]]])
    # [2, 4]
    anchors = tf.constant([[0.25, 0.25, 0.75, 0.75], [-0.5, 0.25, 0.0, 0.75]])
    iou_layer = IOUSimilarity()
    similarity = iou_layer(gt_boxes, anchors)
    expected_out = (
        np.asarray([[1 / 7, 0], [0, 1 / 3]]).astype(np.float32).reshape((1, 2, 2))
    )
    np.testing.assert_allclose(expected_out, similarity)


def test_iou_batched_gt_multi_anchor_normalized_coordinate():
    # batch_size = 2, n_gt_boxes = 1
    # [2, 1, 4]
    gt_boxes = tf.constant([[[0.0, 0.5, 0.5, 1.0]], [[-0.25, 0.25, 0.25, 0.75]]])
    # [2, 4]
    anchors = tf.constant([[0.25, 0.25, 0.75, 0.75], [-0.5, 0.25, 0.0, 0.75]])
    iou_layer = IOUSimilarity()
    similarity = iou_layer(gt_boxes, anchors)
    expected_out = (
        np.asarray([[1 / 7, 0], [0, 1 / 3]]).astype(np.float32).reshape((2, 1, 2))
    )
    np.testing.assert_allclose(expected_out, similarity)


def test_iou_batched_gt_batched_anchor_normalized_coordinate():
    # batch_size = 2, n_gt_boxes = 1
    # [2, 1, 4]
    gt_boxes = tf.constant([[[0.0, 0.5, 0.5, 1.0]], [[-0.25, 0.25, 0.25, 0.75]]])
    # [2, 1, 4]
    anchors = tf.constant([[[0.25, 0.25, 0.75, 0.75]], [[-0.5, 0.25, 0.0, 0.75]]])
    iou_layer = IOUSimilarity()
    similarity = iou_layer(gt_boxes, anchors)
    expected_out = np.asarray([[1 / 7], [1 / 3]]).astype(np.float32).reshape((2, 1, 1))
    np.testing.assert_allclose(expected_out, similarity)


def test_iou_large():
    # [2, 4]
    gt_boxes = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
    # [3, 4]
    anchors = tf.constant(
        [[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0], [0.0, 0.0, 20.0, 20.0]]
    )
    iou_layer = IOUSimilarity()
    similarity = iou_layer(gt_boxes, anchors)
    expected_out = np.asarray([[2 / 16, 0, 6 / 400], [1 / 16, 0.0, 5 / 400]]).astype(
        np.float32
    )
    np.testing.assert_allclose(expected_out, similarity)


def test_ragged_gt_boxes_multi_anchor_absolute_coordinate():
    # [2, ragged, 4]
    gt_boxes = tf.ragged.constant(
        [[[0, 2, 2, 4]], [[-1, 1, 1, 3], [-1, 1, 2, 3]]], ragged_rank=1
    )
    # [2, 4]
    anchors = tf.constant([[1, 1, 3, 3], [-2, 1, 0, 3]])
    iou_layer = IOUSimilarity()
    similarity = iou_layer(gt_boxes, anchors)
    expected_out = tf.ragged.constant([[[1 / 7, 0.0]], [[0.0, 1 / 3], [1 / 4, 1 / 4]]])
    np.testing.assert_allclose(expected_out.values.numpy(), similarity.values.numpy())


def test_ragged_gt_boxes_multi_anchor_normalized_coordinate():
    # [2, ragged, 4]
    gt_boxes = tf.ragged.constant(
        [[[0.0, 0.5, 0.5, 1.0]], [[-0.25, 0.25, 0.25, 0.75], [-0.25, 0.25, 0.5, 0.75]]],
        ragged_rank=1,
    )
    # [2, 4]
    anchors = tf.constant([[0.25, 0.25, 0.75, 0.75], [-0.5, 0.25, 0.0, 0.75]])
    iou_layer = IOUSimilarity()
    similarity = iou_layer(gt_boxes, anchors)
    expected_out = tf.ragged.constant([[[1 / 7, 0.0]], [[0.0, 1 / 3], [1 / 4, 1 / 4]]])
    np.testing.assert_allclose(expected_out.values.numpy(), similarity.values.numpy())


def test_ragged_gt_boxes_batched_anchor_normalized_coordinate():
    # [2, ragged, 4]
    gt_boxes = tf.ragged.constant(
        [[[0.0, 0.5, 0.5, 1.0]], [[-0.25, 0.25, 0.25, 0.75], [-0.25, 0.25, 0.5, 0.75]]],
        ragged_rank=1,
    )
    # [2, 1, 4]
    anchors = tf.constant([[[0.25, 0.25, 0.75, 0.75]], [[-0.5, 0.25, 0.0, 0.75]]])
    iou_layer = IOUSimilarity()
    similarity = iou_layer(gt_boxes, anchors)
    expected_out = tf.ragged.constant([[[1 / 7]], [[1 / 3], [1 / 4]]])
    np.testing.assert_allclose(expected_out.values.numpy(), similarity.values.numpy())


def test_ragged_gt_boxes_empty_anchor():
    # [2, ragged, 4]
    gt_boxes = tf.ragged.constant(
        [[[0.0, 0.5, 0.5, 1.0]], [[-0.25, 0.25, 0.25, 0.75], [-0.25, 0.25, 0.5, 0.75]]],
        ragged_rank=1,
    )
    # [2, 4]
    anchors = tf.constant([[0.25, 0.25, 0.25, 0.25], [-0.5, 0.25, 0.0, 0.75]])
    iou_layer = IOUSimilarity()
    similarity = iou_layer(gt_boxes, anchors)
    expected_out = tf.ragged.constant([[[0.0, 0.0]], [[0.0, 1 / 3], [0.0, 1 / 4]]])
    np.testing.assert_allclose(expected_out.values.numpy(), similarity.values.numpy())
