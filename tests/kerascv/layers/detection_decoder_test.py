import numpy as np
import tensorflow as tf
from kerascv.layers.detection_decoder import DetectionDecoder


def test_single_class_foreground_with_large_iou_threshold():
    scores_pred = tf.constant(
        [[0.4, 0.6], [0.5, 0.5], [0.1, 0.9], [0.3, 0.7], [0.2, 0.8]]
    )
    batched_scores_pred = tf.expand_dims(scores_pred, axis=0)
    # scores in descending order
    # score 0.9 -- bbox [0.5, 0.5, 1.5, 1.5] -- should be picked
    # score 0.8 -- bbox [0, 1, 1, 2] -- should be picked since its iou with first box is 1/7
    # score 0.7 -- bbox [0, 0, 1, 1] -- should be picked since its iou with first box is 1/7, with second box is 0
    # score 0.6 -- bbox [-0.2, 1, 0.8, 2] -- should not be picked since top_k is 3
    # score 0.5 -- bbox [-2, -2, -1, -1] -- should not be picked since top_k is 3
    boxes_pred = tf.constant(
        [
            [-0.2, 1.0, 0.8, 2.0],
            [-2.0, -2.0, -1.0, -1.0],
            [0.5, 0.5, 1.5, 1.5],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 2.0],
        ]
    )
    batched_boxes_pred = tf.expand_dims(boxes_pred, axis=0)
    decoder = DetectionDecoder(
        score_threshold=0.0, iou_threshold=0.25, top_k=3, max_nms_size=5
    )
    top_k_scores, top_k_boxes = decoder(batched_scores_pred, batched_boxes_pred)
    expected_top_k_scores = (
        np.asarray([[1.0, 0.9], [1.0, 0.8], [1.0, 0.7]])
        .astype(np.float32)
        .reshape((1, 3, 2))
    )
    expected_top_k_boxes = (
        np.asarray([[0.5, 0.5, 1.5, 1.5], [0.0, 1.0, 1.0, 2.0], [0.0, 0.0, 1.0, 1.0]])
        .astype(np.float32)
        .reshape((1, 3, 4))
    )
    np.testing.assert_allclose(expected_top_k_scores, top_k_scores)
    np.testing.assert_allclose(expected_top_k_boxes, top_k_boxes)


def test_single_class_foreground_with_large_iou_threshold_top_2():
    scores_pred = tf.constant(
        [[0.4, 0.6], [0.5, 0.5], [0.1, 0.9], [0.3, 0.7], [0.2, 0.8]]
    )
    batched_scores_pred = tf.expand_dims(scores_pred, axis=0)
    # scores in descending order
    # score 0.9 -- bbox [0.5, 0.5, 1.5, 1.5] -- should be picked
    # score 0.8 -- bbox [0, 1, 1, 2] -- should be picked since its iou with first box is 1/7
    # score 0.7 -- bbox [0, 0, 1, 1] -- should be picked since its iou with first box is 1/7, with second box is 0
    # score 0.6 -- bbox [-0.2, 1, 0.8, 2] -- should not be picked since top_k is 3
    # score 0.5 -- bbox [-2, -2, -1, -1] -- should not be picked since top_k is 3
    boxes_pred = tf.constant(
        [
            [-0.2, 1.0, 0.8, 2.0],
            [-2.0, -2.0, -1.0, -1.0],
            [0.5, 0.5, 1.5, 1.5],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 2.0],
        ]
    )
    batched_boxes_pred = tf.expand_dims(boxes_pred, axis=0)
    decoder = DetectionDecoder(
        score_threshold=0.0, iou_threshold=0.25, top_k=2, max_nms_size=5
    )
    top_k_scores, top_k_boxes = decoder(batched_scores_pred, batched_boxes_pred)
    expected_top_k_scores = (
        np.asarray([[1.0, 0.9], [1.0, 0.8]]).astype(np.float32).reshape((1, 2, 2))
    )
    expected_top_k_boxes = (
        np.asarray([[0.5, 0.5, 1.5, 1.5], [0.0, 1.0, 1.0, 2.0]])
        .astype(np.float32)
        .reshape((1, 2, 4))
    )
    np.testing.assert_allclose(expected_top_k_scores, top_k_scores)
    np.testing.assert_allclose(expected_top_k_boxes, top_k_boxes)


def test_single_class_foreground_with_large_iou_threshold_nms_2():
    scores_pred = tf.constant(
        [[0.4, 0.6], [0.5, 0.5], [0.1, 0.9], [0.3, 0.7], [0.2, 0.8]]
    )
    batched_scores_pred = tf.expand_dims(scores_pred, axis=0)
    # scores in descending order
    # score 0.9 -- bbox [0.5, 0.5, 1.5, 1.5] -- should be picked
    # score 0.8 -- bbox [0, 1, 1, 2] -- should be picked since its iou with first box is 1/7
    # score 0.7 -- bbox [0, 0, 1, 1] -- should not be picked since max_nms_size is 2
    # score 0.6 -- bbox [-0.2, 1, 0.8, 2] -- should not be picked
    # score 0.5 -- bbox [-2, -2, -1, -1] -- should not be picked
    boxes_pred = tf.constant(
        [
            [-0.2, 1.0, 0.8, 2.0],
            [-2.0, -2.0, -1.0, -1.0],
            [0.5, 0.5, 1.5, 1.5],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 2.0],
        ]
    )
    batched_boxes_pred = tf.expand_dims(boxes_pred, axis=0)
    decoder = DetectionDecoder(
        score_threshold=0.0, iou_threshold=0.25, top_k=3, max_nms_size=2
    )
    top_k_scores, top_k_boxes = decoder(batched_scores_pred, batched_boxes_pred)
    # Since top_k > max_nms_size, the last one would be padded zero.
    expected_top_k_scores = (
        np.asarray([[1.0, 0.9], [1.0, 0.8], [0.0, 0.0]])
        .astype(np.float32)
        .reshape((1, 3, 2))
    )
    expected_top_k_boxes = (
        np.asarray([[0.5, 0.5, 1.5, 1.5], [0.0, 1.0, 1.0, 2.0], [0.0, 0.0, 0.0, 0.0]])
        .astype(np.float32)
        .reshape((1, 3, 4))
    )
    np.testing.assert_allclose(expected_top_k_scores, top_k_scores)
    np.testing.assert_allclose(expected_top_k_boxes, top_k_boxes)


def test_single_class_foreground_with_large_iou_threshold_top_5():
    scores_pred = tf.constant(
        [[0.4, 0.6], [0.5, 0.5], [0.1, 0.9], [0.3, 0.7], [0.2, 0.8]]
    )
    batched_scores_pred = tf.expand_dims(scores_pred, axis=0)
    # scores in descending order
    # score 0.9 -- bbox [0.5, 0.5, 1.5, 1.5] -- should be picked
    # score 0.8 -- bbox [0, 1, 1, 2] -- should be picked since its iou with first box is 1/7
    # score 0.7 -- bbox [0, 0, 1, 1] -- should be picked since its iou with first box is 1/7, with second box is 0
    # score 0.6 -- bbox [-0.2, 1, 0.8, 2] -- should not be picked since its iou with 2nd box is 0.66
    # score 0.5 -- bbox [-2, -2, -1, -1] -- should be picked
    boxes_pred = tf.constant(
        [
            [-0.2, 1.0, 0.8, 2.0],
            [-2.0, -2.0, -1.0, -1.0],
            [0.5, 0.5, 1.5, 1.5],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 2.0],
        ]
    )
    batched_boxes_pred = tf.expand_dims(boxes_pred, axis=0)
    decoder = DetectionDecoder(
        score_threshold=0.0, iou_threshold=0.25, top_k=5, max_nms_size=5
    )
    top_k_scores, top_k_boxes = decoder(batched_scores_pred, batched_boxes_pred)
    expected_top_k_scores = (
        np.asarray([[1.0, 0.9], [1.0, 0.8], [1.0, 0.7], [1.0, 0.5], [0.0, 0.0]])
        .astype(np.float32)
        .reshape((1, 5, 2))
    )
    expected_top_k_boxes = (
        np.asarray(
            [
                [0.5, 0.5, 1.5, 1.5],
                [0.0, 1.0, 1.0, 2.0],
                [0.0, 0.0, 1.0, 1.0],
                [-2.0, -2.0, -1.0, -1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        .astype(np.float32)
        .reshape((1, 5, 4))
    )
    np.testing.assert_allclose(expected_top_k_scores, top_k_scores)
    np.testing.assert_allclose(expected_top_k_boxes, top_k_boxes)


def test_single_class_foreground_with_large_iou_threshold_top_5_normalized():
    scores_pred = tf.constant(
        [[0.4, 0.6], [0.5, 0.5], [0.1, 0.9], [0.3, 0.7], [0.2, 0.8]]
    )
    batched_scores_pred = tf.expand_dims(scores_pred, axis=0)
    # scores in descending order
    # score 0.9 -- bbox [0.5, 0.5, 1.5, 1.5] -- should be picked
    # score 0.8 -- bbox [0, 1, 1, 2] -- should be picked since its iou with first box is 1/7
    # score 0.7 -- bbox [0, 0, 1, 1] -- should be picked since its iou with first box is 1/7, with second box is 0
    # score 0.6 -- bbox [-0.2, 1, 0.8, 2] -- should not be picked since its iou with 2nd box is 0.66
    # score 0.5 -- bbox [-2, -2, -1, -1] -- should be picked
    # all normaliezd by 2.
    boxes_pred = tf.constant(
        [
            [-0.1, 0.5, 0.4, 1.0],
            [-1.0, -1.0, -0.5, -0.5],
            [0.25, 0.25, 0.75, 0.75],
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.5, 0.5, 1.0],
        ]
    )
    batched_boxes_pred = tf.expand_dims(boxes_pred, axis=0)
    decoder = DetectionDecoder(
        score_threshold=0.0, iou_threshold=0.25, top_k=5, max_nms_size=5
    )
    top_k_scores, top_k_boxes = decoder(batched_scores_pred, batched_boxes_pred)
    expected_top_k_scores = (
        np.asarray([[1.0, 0.9], [1.0, 0.8], [1.0, 0.7], [1.0, 0.5], [0.0, 0.0]])
        .astype(np.float32)
        .reshape((1, 5, 2))
    )
    expected_top_k_boxes = (
        np.asarray(
            [
                [0.25, 0.25, 0.75, 0.75],
                [0.0, 0.5, 0.5, 1.0],
                [0.0, 0.0, 0.5, 0.5],
                [-1.0, -1.0, -0.5, -0.5],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        .astype(np.float32)
        .reshape((1, 5, 4))
    )
    np.testing.assert_allclose(expected_top_k_scores, top_k_scores)
    np.testing.assert_allclose(expected_top_k_boxes, top_k_boxes)


def test_single_class_foreground_select_all():
    scores_pred = tf.constant(
        [[0.4, 0.6], [0.5, 0.5], [0.1, 0.9], [0.3, 0.7], [0.2, 0.8]]
    )
    batched_scores_pred = tf.expand_dims(scores_pred, axis=0)
    # scores in descending order
    # score 0.9 -- bbox [0.5, 0.5, 1.5, 1.5] -- should be picked
    # score 0.8 -- bbox [0, 1, 1, 2] -- should be picked since its iou with first box is 1/7
    # score 0.7 -- bbox [0, 0, 1, 1] -- should be picked since its iou with first box is 1/7, with second box is 0
    # score 0.6 -- bbox [-0.2, 1, 0.8, 2] -- should be picked since its iou with 2nd box is 0.66
    # score 0.5 -- bbox [-2, -2, -1, -1] -- should be picked
    boxes_pred = tf.constant(
        [
            [-0.2, 1.0, 0.8, 2.0],
            [-2.0, -2.0, -1.0, -1.0],
            [0.5, 0.5, 1.5, 1.5],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 2.0],
        ]
    )
    batched_boxes_pred = tf.expand_dims(boxes_pred, axis=0)
    decoder = DetectionDecoder(
        score_threshold=0.0, iou_threshold=0.7, top_k=5, max_nms_size=5
    )
    top_k_scores, top_k_boxes = decoder(batched_scores_pred, batched_boxes_pred)
    expected_top_k_scores = (
        np.asarray([[1.0, 0.9], [1.0, 0.8], [1.0, 0.7], [1.0, 0.6], [1.0, 0.5]])
        .astype(np.float32)
        .reshape((1, 5, 2))
    )
    expected_top_k_boxes = (
        np.asarray(
            [
                [0.5, 0.5, 1.5, 1.5],
                [0.0, 1.0, 1.0, 2.0],
                [0.0, 0.0, 1.0, 1.0],
                [-0.2, 1.0, 0.8, 2.0],
                [-2.0, -2.0, -1.0, -1.0],
            ]
        )
        .astype(np.float32)
        .reshape((1, 5, 4))
    )
    np.testing.assert_allclose(expected_top_k_scores, top_k_scores)
    np.testing.assert_allclose(expected_top_k_boxes, top_k_boxes)


def test_single_class_foreground_large_score_threshold():
    scores_pred = tf.constant(
        [[0.4, 0.6], [0.5, 0.5], [0.1, 0.9], [0.3, 0.7], [0.2, 0.8]]
    )
    batched_scores_pred = tf.expand_dims(scores_pred, axis=0)
    # scores in descending order
    # score 0.9 -- bbox [0.5, 0.5, 1.5, 1.5] -- should be picked
    # score 0.8 -- bbox [0, 1, 1, 2] -- should be picked since its iou with first box is 1/7
    # score 0.7 -- bbox [0, 0, 1, 1] -- should be picked since its iou with first box is 1/7, with second box is 0
    # score 0.6 -- bbox [-0.2, 1, 0.8, 2] -- should not be picked since its iou with 2nd box is 0.66
    # score 0.5 -- bbox [-2, -2, -1, -1] -- should be picked
    boxes_pred = tf.constant(
        [
            [-0.2, 1.0, 0.8, 2.0],
            [-2.0, -2.0, -1.0, -1.0],
            [0.5, 0.5, 1.5, 1.5],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 2.0],
        ]
    )
    batched_boxes_pred = tf.expand_dims(boxes_pred, axis=0)
    decoder = DetectionDecoder(
        score_threshold=0.69, iou_threshold=0.7, top_k=5, max_nms_size=5
    )
    top_k_scores, top_k_boxes = decoder(batched_scores_pred, batched_boxes_pred)
    expected_top_k_scores = (
        np.asarray([[1.0, 0.9], [1.0, 0.8], [1.0, 0.7], [0.0, 0.0], [0.0, 0.0]])
        .astype(np.float32)
        .reshape((1, 5, 2))
    )
    expected_top_k_boxes = (
        np.asarray(
            [
                [0.5, 0.5, 1.5, 1.5],
                [0.0, 1.0, 1.0, 2.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        .astype(np.float32)
        .reshape((1, 5, 4))
    )
    np.testing.assert_allclose(expected_top_k_scores, top_k_scores)
    np.testing.assert_allclose(expected_top_k_boxes, top_k_boxes)


def test_single_class_foreground_large_score_threshold_nms_2():
    scores_pred = tf.constant(
        [[0.4, 0.6], [0.5, 0.5], [0.1, 0.9], [0.3, 0.7], [0.2, 0.8]]
    )
    batched_scores_pred = tf.expand_dims(scores_pred, axis=0)
    # scores in descending order
    # score 0.9 -- bbox [0.5, 0.5, 1.5, 1.5] -- should be picked
    # score 0.8 -- bbox [0, 1, 1, 2] -- should be picked since its iou with first box is 1/7
    # score 0.7 -- bbox [0, 0, 1, 1] -- should not be picked since max_nms_size is 2
    # score 0.6 -- bbox [-0.2, 1, 0.8, 2] -- should not be picked since its iou with 2nd box is 0.66
    # score 0.5 -- bbox [-2, -2, -1, -1] -- should be picked
    boxes_pred = tf.constant(
        [
            [-0.2, 1.0, 0.8, 2.0],
            [-2.0, -2.0, -1.0, -1.0],
            [0.5, 0.5, 1.5, 1.5],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 2.0],
        ]
    )
    batched_boxes_pred = tf.expand_dims(boxes_pred, axis=0)
    decoder = DetectionDecoder(
        score_threshold=0.69, iou_threshold=0.7, top_k=5, max_nms_size=2
    )
    top_k_scores, top_k_boxes = decoder(batched_scores_pred, batched_boxes_pred)
    expected_top_k_scores = (
        np.asarray([[1.0, 0.9], [1.0, 0.8], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        .astype(np.float32)
        .reshape((1, 5, 2))
    )
    expected_top_k_boxes = (
        np.asarray(
            [
                [0.5, 0.5, 1.5, 1.5],
                [0.0, 1.0, 1.0, 2.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        .astype(np.float32)
        .reshape((1, 5, 4))
    )
    np.testing.assert_allclose(expected_top_k_scores, top_k_scores)
    np.testing.assert_allclose(expected_top_k_boxes, top_k_boxes)


def test_two_classes_foreground_with_large_iou_threshold():
    scores_pred = tf.constant(
        [
            [0.4, 0.6, 0.9],
            [0.5, 0.5, 0.7],
            [0.1, 0.9, 0.5],
            [0.3, 0.7, 0.2],
            [0.2, 0.8, 1.2],
        ]
    )
    batched_scores_pred = tf.expand_dims(scores_pred, axis=0)
    # class 1
    # scores in descending order
    # score 0.9 -- bbox [0.5, 0.5, 1.5, 1.5] -- should be picked and compared with 2nd class
    # score 0.8 -- bbox [0, 1, 1, 2] -- should be picked and compared with 2nd class
    # score 0.7 -- bbox [0, 0, 1, 1] -- should be picked and compared with 2nd class
    # score 0.6 -- bbox [-0.2, 1, 0.8, 2] -- should not be picked since max_nms_size is 3
    # score 0.5 -- bbox [-2, -2, -1, -1] -- should not be picked since max_nms_size is 3

    # class 2
    # scores in descending order
    # score 1.2 -- bbox [0, 1, 1, 2] -- should be picked and compared with 1st class
    # score 0.9 -- bbox [-0.2, 1., 0.8, 2.] -- should not be picked since its iou is 2/3 with first bbox
    # score 0.7 -- bbox [-2, -2, -1, -1] -- should be picked and compared with 1st class
    # score 0.5 -- bbox [0.5, 0.5, 1.5, 1.5] -- should be picked and compared since its iou is 1/7 with first bbox
    # score 0.2 -- bbox [0, 0, 1, 1] -- should not be picked since max_nms_size is 3

    boxes_pred = tf.constant(
        [
            [-0.2, 1.0, 0.8, 2.0],
            [-2.0, -2.0, -1.0, -1.0],
            [0.5, 0.5, 1.5, 1.5],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 2.0],
        ]
    )
    batched_boxes_pred = tf.expand_dims(boxes_pred, axis=0)
    decoder = DetectionDecoder(
        score_threshold=0.0, iou_threshold=0.25, top_k=5, max_nms_size=3
    )
    top_k_scores, top_k_boxes = decoder(batched_scores_pred, batched_boxes_pred)
    expected_top_k_scores = (
        np.asarray([[2.0, 1.2], [1.0, 0.9], [1.0, 0.8], [1.0, 0.7], [2.0, 0.7]])
        .astype(np.float32)
        .reshape((1, 5, 2))
    )
    expected_top_k_boxes = (
        np.asarray(
            [
                [0.0, 1.0, 1.0, 2.0],
                [0.5, 0.5, 1.5, 1.5],
                [0.0, 1.0, 1.0, 2.0],
                [0.0, 0.0, 1.0, 1.0],
                [-2.0, -2.0, -1.0, -1.0],
            ]
        )
        .astype(np.float32)
        .reshape((1, 5, 4))
    )
    np.testing.assert_allclose(expected_top_k_scores, top_k_scores)
    np.testing.assert_allclose(expected_top_k_boxes, top_k_boxes)


def test_two_classes_foreground_with_larger_iou_threshold_top_5():
    scores_pred = tf.constant(
        [
            [0.4, 0.6, 0.9],
            [0.5, 0.5, 0.7],
            [0.1, 0.9, 0.5],
            [0.3, 0.7, 0.2],
            [0.2, 0.8, 1.2],
        ]
    )
    batched_scores_pred = tf.expand_dims(scores_pred, axis=0)
    # class 1
    # scores in descending order
    # score 0.9 -- bbox [0.5, 0.5, 1.5, 1.5] -- should be picked and compared with 2nd class
    # score 0.8 -- bbox [0, 1, 1, 2] -- should be picked and compared with 2nd class
    # score 0.7 -- bbox [0, 0, 1, 1] -- should be picked and compared with 2nd class
    # score 0.6 -- bbox [-0.2, 1, 0.8, 2] -- should not be picked since max_nms_size is 3
    # score 0.5 -- bbox [-2, -2, -1, -1] -- should not be picked since max_nms_size is 3

    # class 2
    # scores in descending order
    # score 1.2 -- bbox [0, 1, 1, 2] -- should be picked and compared with 1st class
    # score 0.9 -- bbox [-0.2, 1., 0.8, 2.] -- should be picked and compared since its iou is 2/3 with first bbox
    # score 0.7 -- bbox [-2, -2, -1, -1] -- should be picked and compared with 1st class
    # score 0.5 -- bbox [0.5, 0.5, 1.5, 1.5] -- should be picked and compared since its iou is 1/7 with first bbox
    # score 0.2 -- bbox [0, 0, 1, 1] -- should not be picked since max_nms_size is 3

    boxes_pred = tf.constant(
        [
            [-0.2, 1.0, 0.8, 2.0],
            [-2.0, -2.0, -1.0, -1.0],
            [0.5, 0.5, 1.5, 1.5],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 2.0],
        ]
    )
    batched_boxes_pred = tf.expand_dims(boxes_pred, axis=0)
    decoder = DetectionDecoder(
        score_threshold=0.0, iou_threshold=0.7, top_k=5, max_nms_size=3
    )
    top_k_scores, top_k_boxes = decoder(batched_scores_pred, batched_boxes_pred)
    expected_top_k_scores = (
        np.asarray([[2.0, 1.2], [1.0, 0.9], [2.0, 0.9], [1.0, 0.8], [1.0, 0.7]])
        .astype(np.float32)
        .reshape((1, 5, 2))
    )
    expected_top_k_boxes = (
        np.asarray(
            [
                [0.0, 1.0, 1.0, 2.0],
                [0.5, 0.5, 1.5, 1.5],
                [-0.2, 1.0, 0.8, 2.0],
                [0.0, 1.0, 1.0, 2.0],
                [0.0, 0.0, 1.0, 1.0],
            ]
        )
        .astype(np.float32)
        .reshape((1, 5, 4))
    )
    np.testing.assert_allclose(expected_top_k_scores, top_k_scores)
    np.testing.assert_allclose(expected_top_k_boxes, top_k_boxes)


def test_single_class_batched_foreground_with_large_iou_threshold():
    batched_scores_pred = tf.constant(
        [
            [[0.4, 0.6], [0.5, 0.5], [0.1, 0.9], [0.3, 0.7], [0.2, 0.8]],
            [[0.4, 0.9], [0.5, 0.7], [0.1, 0.5], [0.3, 0.2], [0.2, 1.2]],
        ]
    )
    # batch 1
    # scores in descending order
    # score 0.9 -- bbox [0.5, 0.5, 1.5, 1.5] -- should be picked
    # score 0.8 -- bbox [0, 1, 1, 2] -- should not be picked
    # score 0.7 -- bbox [0, 0, 1, 1] -- should be picked
    # score 0.6 -- bbox [-0.2, 1, 0.8, 2] -- should be picked
    # score 0.5 -- bbox [-2, -2, -1, -1] -- should not be picked

    # batch 2
    # scores in descending order
    # score 1.2 -- bbox [0, 1, 1, 2] -- should be picked
    # score 0.9 -- bbox [-0.2, 1., 0.8, 2.] -- should not be picked since its iou is 2/3 with first bbox
    # score 0.7 -- bbox [-2, -2, -1, -1] -- should be picked
    # score 0.5 -- bbox [0.5, 0.5, 1.5, 1.5] -- should not be picked since its iou is 1/7 with first bbox
    # score 0.2 -- bbox [0, 0, 1, 1] -- should not be picked

    boxes_pred = tf.constant(
        [
            [-0.2, 1.0, 0.8, 2.0],
            [-2.0, -2.0, -1.0, -1.0],
            [0.5, 0.5, 1.5, 1.5],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 2.0],
        ]
    )
    batched_boxes_pred = tf.tile(tf.expand_dims(boxes_pred, axis=0), [2, 1, 1])
    decoder = DetectionDecoder(
        score_threshold=0.0, iou_threshold=1 / 7, top_k=3, max_nms_size=5
    )
    top_k_scores, top_k_boxes = decoder(batched_scores_pred, batched_boxes_pred)
    print("top k scores {}".format(top_k_scores))
    expected_top_k_scores = np.asarray(
        [[[1.0, 0.9], [1.0, 0.6], [1.0, 0.5]], [[1.0, 1.2], [1.0, 0.7], [1.0, 0.2]]]
    ).astype(np.float32)
    expected_top_k_boxes = np.asarray(
        [
            [[0.5, 0.5, 1.5, 1.5], [-0.2, 1, 0.8, 2], [-2, -2, -1, -1]],
            [[0, 1, 1, 2], [-2, -2, -1, -1], [0, 0, 1, 1]],
        ]
    ).astype(np.float32)
    np.testing.assert_allclose(expected_top_k_scores, top_k_scores)
    np.testing.assert_allclose(expected_top_k_boxes, top_k_boxes)


def test_two_classes_foreground_with_larger_iou_threshold_top_5():
    scores_pred = tf.constant(
        [
            [0.4, 0.6, 0.9],
            [0.5, 0.5, 0.7],
            [0.1, 0.9, 0.5],
            [0.3, 0.7, 0.2],
            [0.2, 0.8, 1.2],
        ]
    )
    batched_scores_pred = tf.expand_dims(scores_pred, axis=0)
    # class 1
    # scores in descending order
    # score 0.9 -- bbox [0.5, 0.5, 1.5, 1.5] -- should be picked and compared with 2nd class
    # score 0.8 -- bbox [0, 1, 1, 2] -- should be picked and compared with 2nd class
    # score 0.7 -- bbox [0, 0, 1, 1] -- should be picked and compared with 2nd class
    # score 0.6 -- bbox [-0.2, 1, 0.8, 2] -- should not be picked since max_nms_size is 3
    # score 0.5 -- bbox [-2, -2, -1, -1] -- should not be picked since max_nms_size is 3

    # class 2
    # scores in descending order
    # score 1.2 -- bbox [0, 1, 1, 2] -- should be picked and compared with 1st class
    # score 0.9 -- bbox [-0.2, 1., 0.8, 2.] -- should be picked and compared since its iou is 2/3 with first bbox
    # score 0.7 -- bbox [-2, -2, -1, -1] -- should be picked and compared with 1st class
    # score 0.5 -- bbox [0.5, 0.5, 1.5, 1.5] -- should be picked and compared since its iou is 1/7 with first bbox
    # score 0.2 -- bbox [0, 0, 1, 1] -- should not be picked since max_nms_size is 3

    boxes_pred = tf.constant(
        [
            [-0.2, 1.0, 0.8, 2.0],
            [-2.0, -2.0, -1.0, -1.0],
            [0.5, 0.5, 1.5, 1.5],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 2.0],
        ]
    )
    batched_boxes_pred = tf.expand_dims(boxes_pred, axis=0)
    decoder = DetectionDecoder(
        score_threshold=0.0, iou_threshold=0.7, top_k=5, max_nms_size=3
    )
    top_k_scores, top_k_boxes = decoder(batched_scores_pred, batched_boxes_pred)
    expected_top_k_scores = (
        np.asarray([[2.0, 1.2], [1.0, 0.9], [2.0, 0.9], [1.0, 0.8], [1.0, 0.7]])
        .astype(np.float32)
        .reshape((1, 5, 2))
    )
    expected_top_k_boxes = (
        np.asarray(
            [
                [0.0, 1.0, 1.0, 2.0],
                [0.5, 0.5, 1.5, 1.5],
                [-0.2, 1.0, 0.8, 2.0],
                [0.0, 1.0, 1.0, 2.0],
                [0.0, 0.0, 1.0, 1.0],
            ]
        )
        .astype(np.float32)
        .reshape((1, 5, 4))
    )
    np.testing.assert_allclose(expected_top_k_scores, top_k_scores)
    np.testing.assert_allclose(expected_top_k_boxes, top_k_boxes)


def test_single_class_foreground_with_small_iou_threshold():
    scores_pred = tf.constant(
        [[0.4, 0.6], [0.5, 0.5], [0.1, 0.9], [0.3, 0.7], [0.2, 0.8]]
    )
    batched_scores_pred = tf.expand_dims(scores_pred, axis=0)
    # scores in descending order
    # score 0.9 -- bbox [0.5, 0.5, 1.5, 1.5] -- should be picked
    # score 0.8 -- bbox [0, 1, 1, 2] -- should not be picked since its iou with first box is 1/7
    # score 0.7 -- bbox [0, 0, 1, 1] -- should not be picked since its iou with first box is 1/7, with second box is 0
    # score 0.6 -- bbox [-0.2, 1, 0.8, 2] -- should be picked since its iou with first box is 0.15 / (2. - 0.15) < 1/7
    # score 0.5 -- bbox [-2, -2, -1, -1] -- should be picked since since it doesn't intersect with any other box
    boxes_pred = tf.constant(
        [
            [-0.2, 1.0, 0.8, 2.0],
            [-2.0, -2.0, -1.0, -1.0],
            [0.5, 0.5, 1.5, 1.5],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 2.0],
        ]
    )
    batched_boxes_pred = tf.expand_dims(boxes_pred, axis=0)
    decoder = DetectionDecoder(
        score_threshold=0.0, iou_threshold=1 / 7, top_k=3, max_nms_size=5
    )
    top_k_scores, top_k_boxes = decoder(batched_scores_pred, batched_boxes_pred)
    expected_top_k_scores = (
        np.asarray([[1.0, 0.9], [1.0, 0.6], [1.0, 0.5]])
        .astype(np.float32)
        .reshape((1, 3, 2))
    )
    expected_top_k_boxes = (
        np.asarray(
            [[0.5, 0.5, 1.5, 1.5], [-0.2, 1.0, 0.8, 2.0], [-2.0, -2.0, -1.0, -1.0]]
        )
        .astype(np.float32)
        .reshape((1, 3, 4))
    )
    np.testing.assert_allclose(expected_top_k_scores, top_k_scores)
    np.testing.assert_allclose(expected_top_k_boxes, top_k_boxes)


def test_single_class_foreground_with_small_iou_threshold_top_2():
    scores_pred = tf.constant(
        [[0.4, 0.6], [0.5, 0.5], [0.1, 0.9], [0.3, 0.7], [0.2, 0.8]]
    )
    batched_scores_pred = tf.expand_dims(scores_pred, axis=0)
    # scores in descending order
    # score 0.9 -- bbox [0.5, 0.5, 1.5, 1.5] -- should be picked
    # score 0.8 -- bbox [0, 1, 1, 2] -- should not be picked since its iou with first box is 1/7
    # score 0.7 -- bbox [0, 0, 1, 1] -- should not be picked since its iou with first box is 1/7, with second box is 0
    # score 0.6 -- bbox [-0.2, 1, 0.8, 2] -- should be picked since its iou with first box is 0.15 / (2. - 0.15) < 1/7
    # score 0.5 -- bbox [-2, -2, -1, -1] -- should be picked since since it doesn't intersect with any other box
    boxes_pred = tf.constant(
        [
            [-0.2, 1.0, 0.8, 2.0],
            [-2.0, -2.0, -1.0, -1.0],
            [0.5, 0.5, 1.5, 1.5],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 2.0],
        ]
    )
    batched_boxes_pred = tf.expand_dims(boxes_pred, axis=0)
    decoder = DetectionDecoder(
        score_threshold=0.0, iou_threshold=1 / 7, top_k=2, max_nms_size=5
    )
    top_k_scores, top_k_boxes = decoder(batched_scores_pred, batched_boxes_pred)
    expected_top_k_scores = (
        np.asarray([[1.0, 0.9], [1.0, 0.6]]).astype(np.float32).reshape((1, 2, 2))
    )
    expected_top_k_boxes = (
        np.asarray([[0.5, 0.5, 1.5, 1.5], [-0.2, 1.0, 0.8, 2.0]])
        .astype(np.float32)
        .reshape((1, 2, 4))
    )
    np.testing.assert_allclose(expected_top_k_scores, top_k_scores)
    np.testing.assert_allclose(expected_top_k_boxes, top_k_boxes)


def test_single_class_foreground_with_tiny_iou_threshold():
    scores_pred = tf.constant(
        [[0.4, 0.6], [0.5, 0.5], [0.1, 0.9], [0.3, 0.7], [0.2, 0.8]]
    )
    batched_scores_pred = tf.expand_dims(scores_pred, axis=0)
    # scores in descending order
    # score 0.9 -- bbox [0.5, 0.5, 1.5, 1.5] -- should be picked
    # score 0.8 -- bbox [0, 1, 1, 2] -- should not be picked since its iou with first box is 1/7
    # score 0.7 -- bbox [0, 0, 1, 1] -- should not be picked since its iou with first box is 1/7, with second box is 0
    # score 0.6 -- bbox [-0.2, 1, 0.8, 2] -- should not be picked since its iou with first box is 0.15 / (2. - 0.15)
    # score 0.5 -- bbox [-2, -2, -1, -1] -- should be picked since since it doesn't intersect with any other box
    boxes_pred = tf.constant(
        [
            [-0.2, 1.0, 0.8, 2.0],
            [-2.0, -2.0, -1.0, -1.0],
            [0.5, 0.5, 1.5, 1.5],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 2.0],
        ]
    )
    batched_boxes_pred = tf.expand_dims(boxes_pred, axis=0)
    decoder = DetectionDecoder(
        score_threshold=0.0, iou_threshold=0.15 / 1.85, top_k=3, max_nms_size=5
    )
    top_k_scores, top_k_boxes = decoder(batched_scores_pred, batched_boxes_pred)
    expected_top_k_scores = (
        np.asarray([[1.0, 0.9], [1.0, 0.5], [0.0, 0.0]])
        .astype(np.float32)
        .reshape((1, 3, 2))
    )
    expected_top_k_boxes = (
        np.asarray(
            [[0.5, 0.5, 1.5, 1.5], [-2.0, -2.0, -1.0, -1.0], [0.0, 0.0, 0.0, 0.0]]
        )
        .astype(np.float32)
        .reshape((1, 3, 4))
    )
    np.testing.assert_allclose(expected_top_k_scores, top_k_scores)
    np.testing.assert_allclose(expected_top_k_boxes, top_k_boxes)


def test_single_class_foreground_with_tiny_iou_threshold_top_2():
    scores_pred = tf.constant(
        [[0.4, 0.6], [0.5, 0.5], [0.1, 0.9], [0.3, 0.7], [0.2, 0.8]]
    )
    batched_scores_pred = tf.expand_dims(scores_pred, axis=0)
    # scores in descending order
    # score 0.9 -- bbox [0.5, 0.5, 1.5, 1.5] -- should be picked
    # score 0.8 -- bbox [0, 1, 1, 2] -- should not be picked since its iou with first box is 1/7
    # score 0.7 -- bbox [0, 0, 1, 1] -- should not be picked since its iou with first box is 1/7, with second box is 0
    # score 0.6 -- bbox [-0.2, 1, 0.8, 2] -- should not be picked since its iou with first box is 0.15 / (2. - 0.15)
    # score 0.5 -- bbox [-2, -2, -1, -1] -- should be picked since since it doesn't intersect with any other box
    boxes_pred = tf.constant(
        [
            [-0.2, 1.0, 0.8, 2.0],
            [-2.0, -2.0, -1.0, -1.0],
            [0.5, 0.5, 1.5, 1.5],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 2.0],
        ]
    )
    batched_boxes_pred = tf.expand_dims(boxes_pred, axis=0)
    decoder = DetectionDecoder(
        score_threshold=0.0, iou_threshold=0.15 / 1.85, top_k=2, max_nms_size=5
    )
    top_k_scores, top_k_boxes = decoder(batched_scores_pred, batched_boxes_pred)
    expected_top_k_scores = (
        np.asarray([[1.0, 0.9], [1.0, 0.5]]).astype(np.float32).reshape((1, 2, 2))
    )
    expected_top_k_boxes = (
        np.asarray([[0.5, 0.5, 1.5, 1.5], [-2.0, -2.0, -1.0, -1.0]])
        .astype(np.float32)
        .reshape((1, 2, 4))
    )
    np.testing.assert_allclose(expected_top_k_scores, top_k_scores)
    np.testing.assert_allclose(expected_top_k_boxes, top_k_boxes)
