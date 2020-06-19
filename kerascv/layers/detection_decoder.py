import tensorflow as tf


class DetectionDecoder(tf.keras.layers.Layer):
    """Decode the detection with per class non-max-suppression and top_k sorting.

    This Decoder always assume the predicted bounding boxes are in corners format, it applies NMS per class and get
    only `max_nms_size` candidates, and then sort them across all classes and get only `top_k` candidates.
    It returns the top k scores as [batch_size, top_k, 2] where the first column is the class id, and the second column
    is score, it also returns the top k bounding boxes [batch_size, top_k, 4].

    # Attributes:
        score_threshold: A single float to filter out bounding boxes with scores less than or equal to it.
        iou_threshold: A single float to filter out bounding boxes with iou between previously greedily selected
            bounding boxes less than or equal to it.
        top_k: A single int, the final top k bounding boxes per image.
        max_nms_size: A single int, the maximum number of bounding box after non-max-suppression per class, per image.
            If `max_nms_size` * num_classes < `top_k` then the rest would be padded with zeros.
    """

    # TODO: Consider having an option for class agnostic NMS.
    def __init__(
        self, score_threshold, iou_threshold, top_k, max_nms_size, name=None, **kwargs
    ):
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.top_k = top_k
        self.max_nms_size = max_nms_size
        super(DetectionDecoder, self).__init__(name=name, **kwargs)

    # boxes_pred: [batch_size, n_boxes, 4], in corners format, w/ or w/o normalize.
    # scores_pred: [batch_size, n_boxes, n_classes]
    # The assumption is 0 is always background class.
    def call(self, scores_pred, boxes_pred):
        n_classes = tf.shape(scores_pred)[2]
        top_k = tf.constant(self.top_k)
        max_nms_size = tf.constant(self.max_nms_size)
        score_threshold = tf.cast(self.score_threshold, tf.float32)
        iou_threshold = tf.cast(self.iou_threshold, tf.float32)

        # [n_boxes, n_classes], [n_boxes, 4]
        def per_sample_decode(per_sample_scores_pred, per_sample_box_pred):
            def per_class_nms(class_ind):
                # [n_boxes]
                scores = per_sample_scores_pred[..., class_ind]
                selected_indices = tf.image.non_max_suppression(
                    boxes=per_sample_box_pred,
                    scores=scores,
                    max_output_size=max_nms_size,
                    iou_threshold=iou_threshold,
                    score_threshold=score_threshold,
                )
                # [<=max_nms_size, 1]
                nms_per_cls_scores = tf.expand_dims(
                    tf.gather(scores, selected_indices), axis=1
                )
                class_ids = tf.fill(
                    tf.shape(nms_per_cls_scores), value=tf.cast(class_ind, tf.float32)
                )
                # [<=max_nms_size, 2]
                nms_per_cls_scores = tf.concat([class_ids, nms_per_cls_scores], axis=1)
                # [<=max_nms_size, 4]
                nms_per_cls_boxes = tf.gather(per_sample_box_pred, selected_indices)
                padded_nms_per_cls_scores = tf.pad(
                    tensor=nms_per_cls_scores,
                    paddings=[
                        [0, max_nms_size - tf.shape(nms_per_cls_scores)[0]],
                        [0, 0],
                    ],
                    mode="CONSTANT",
                    constant_values=0.0,
                )
                padded_nms_per_cls_boxes = tf.pad(
                    tensor=nms_per_cls_boxes,
                    paddings=[
                        [0, max_nms_size - tf.shape(nms_per_cls_boxes)[0]],
                        [0, 0],
                    ],
                    mode="CONSTANT",
                    constant_values=0.0,
                )
                # [max_nms_size, 2], [max_nms_size, 4]
                return padded_nms_per_cls_scores, padded_nms_per_cls_boxes

            # [n_classes, max_nms_size, 2], [n_classes, max_nms_size, 4]
            padded_nms_scores_per_sample, padded_nms_boxes_per_sample = tf.map_fn(
                fn=lambda i: per_class_nms(i),
                elems=tf.range(1, n_classes),
                dtype=(tf.float32, tf.float32),
                parallel_iterations=32,
                swap_memory=False,
                infer_shape=True,
            )
            # [n_classes * max_nms_size, 2], [n_classes * max_nms_size, 4]
            padded_nms_scores_per_sample = tf.reshape(
                padded_nms_scores_per_sample, (-1, 2)
            )
            padded_nms_boxes_per_sample = tf.reshape(
                padded_nms_boxes_per_sample, (-1, 4)
            )
            padded_nms_shape = tf.shape(padded_nms_scores_per_sample)[0]
            # Autograph cond.
            if top_k > padded_nms_shape:
                padded_nms_scores_per_sample = tf.pad(
                    tensor=padded_nms_scores_per_sample,
                    paddings=[[0, top_k - padded_nms_shape], [0, 0]],
                    mode="CONSTANT",
                    constant_values=0.0,
                )
                padded_nms_boxes_per_sample = tf.pad(
                    tensor=padded_nms_boxes_per_sample,
                    paddings=[[0, top_k - padded_nms_shape], [0, 0]],
                    mode="CONSTANT",
                    constant_values=0.0,
                )

            _, top_k_indices = tf.nn.top_k(
                padded_nms_scores_per_sample[:, 1], k=top_k, sorted=True
            )
            top_k_scores_per_sample = tf.gather(
                params=padded_nms_scores_per_sample, indices=top_k_indices, axis=0
            )
            top_k_boxes_per_sample = tf.gather(
                params=padded_nms_boxes_per_sample, indices=top_k_indices, axis=0
            )
            # [top_k, 2], [top_k, 4]
            return top_k_scores_per_sample, top_k_boxes_per_sample

        # [batch_size, top_k, 2], [batch_size, top_k, 4]
        decoded_scores, decoded_boxes = tf.map_fn(
            fn=lambda x: per_sample_decode(x[0], x[1]),
            elems=(scores_pred, boxes_pred),
            dtype=(tf.float32, tf.float32),
            parallel_iterations=32,
            swap_memory=False,
            infer_shape=True,
        )
        return decoded_scores, decoded_boxes

    def get_config(self):
        config = {
            "scores_threshold": self.scores_threshold,
            "iou_threshold": self.iou_threshold,
            "top_k": self.top_k,
            "max_nms_size": self.max_nms_size,
        }
        base_config = super(DetectionDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
