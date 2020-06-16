import tensorflow as tf


class IOUSimilarity(tf.keras.layers.Layer):
    """Defines a IOUSimilarity that calculates the IOU between ground truth boxes and anchors.

    Calling the layer with `ground_truth_boxes` and `anchors`, `ground_truth_boxes` can be a batched
    `tf.Tensor` or `tf.RaggedTensor`, while `anchors` can be a batched or un-batched `tf.Tensor`.
    """

    def __init__(self, name=None, **kwargs):
        super(IOUSimilarity, self).__init__(name=name, **kwargs)

    def call(self, ground_truth_boxes, anchors):
        # ground_truth_box [n_gt_boxes, box_dim] or [batch_size, n_gt_boxes, box_dim]
        # anchor [n_anchors, box_dim]
        def iou(ground_truth_box, anchor):
            # [n_anchors, 1]
            y_min_anchors, x_min_anchors, y_max_anchors, x_max_anchors = tf.split(
                anchor, num_or_size_splits=4, axis=-1
            )
            # [n_gt_boxes, 1] or [batch_size, n_gt_boxes, 1]
            y_min_gt, x_min_gt, y_max_gt, x_max_gt = tf.split(
                ground_truth_box, num_or_size_splits=4, axis=-1
            )
            # [n_anchors]
            anchor_areas = tf.squeeze(
                (y_max_anchors - y_min_anchors) * (x_max_anchors - x_min_anchors), [1]
            )
            # [n_gt_boxes, 1] or [batch_size, n_gt_boxes, 1]
            gt_areas = (y_max_gt - y_min_gt) * (x_max_gt - x_min_gt)

            # [n_gt_boxes, n_anchors] or [batch_size, n_gt_boxes, n_anchors]
            max_y_min = tf.maximum(y_min_gt, tf.transpose(y_min_anchors))
            min_y_max = tf.minimum(y_max_gt, tf.transpose(y_max_anchors))
            intersect_heights = tf.maximum(
                tf.constant(0, dtype=ground_truth_box.dtype), (min_y_max - max_y_min)
            )

            # [n_gt_boxes, n_anchors] or [batch_size, n_gt_boxes, n_anchors]
            max_x_min = tf.maximum(x_min_gt, tf.transpose(x_min_anchors))
            min_x_max = tf.minimum(x_max_gt, tf.transpose(x_max_anchors))
            intersect_widths = tf.maximum(
                tf.constant(0, dtype=ground_truth_box.dtype), (min_x_max - max_x_min)
            )

            # [n_gt_boxes, n_anchors] or [batch_size, n_gt_boxes, n_anchors]
            intersections = intersect_heights * intersect_widths

            # [n_gt_boxes, n_anchors] or [batch_size, n_gt_boxes, n_anchors]
            unions = gt_areas + anchor_areas - intersections

            return tf.cast(tf.truediv(intersections, unions), tf.float32)

        if isinstance(ground_truth_boxes, tf.RaggedTensor):
            if anchors.shape.ndims == 2:
                return tf.map_fn(
                    lambda x: iou(x, anchors),
                    elems=ground_truth_boxes,
                    parallel_iterations=32,
                    back_prop=False,
                    fn_output_signature=tf.RaggedTensorSpec(
                        dtype=tf.float32, ragged_rank=0
                    ),
                )
            else:
                return tf.map_fn(
                    lambda x: iou(x[0], x[1]),
                    elems=[ground_truth_boxes, anchors],
                    parallel_iterations=32,
                    back_prop=False,
                    fn_output_signature=tf.RaggedTensorSpec(
                        dtype=tf.float32, ragged_rank=0
                    ),
                )
        if anchors.shape.ndims == 2:
            return iou(ground_truth_boxes, anchors)
        elif anchors.shape.ndims == 3:
            return tf.map_fn(
                lambda x: iou(x[0], x[1]),
                elems=[ground_truth_boxes, anchors],
                dtype=tf.float32,
                parallel_iterations=32,
                back_prop=False,
            )
