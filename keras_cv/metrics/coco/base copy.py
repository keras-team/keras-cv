import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.initializers as initializers
from keras_cv import bbox
from keras_cv.metrics.coco import iou as iou_lib
from keras_cv.metrics.coco import util


class COCOBase(keras.metrics.Metric):
    """COCOBase serves as a base for COCORecall and COCOPrecision.

    Args:
        iou_thresholds: defaults to [0.5:0.05:0.95].  Dimension T=len(iou_thresholds), defaults to 10.
        category_ids: no default, users must provide.  K=len(category_ids)
        area_ranges: ranges to consider detections in, defaults to [all, 0-32, 32-96, 96>].
        max_detections: TODO

    Internally the COCOBase class tracks the following values:
    - TruePositives: tf.Tensor with shape [TxKxAxM] precision for every evaluation setting.
    - FalsePositives: tf.Tensor with shape [TxKxAxM] precision for every evaluation setting.
    - GroundTruthBoxes: tf.Tensor with shape [KxA] max recall for every evaluation setting.
    """

    def __init__(
        self,
        iou_thresholds=None,
        category_ids=None,
        area_range=None,
        max_detections=None,
        **kwargs
    ):
        super(COCOBase, self).__init__(**kwargs)
        # Initialize parameter values
        self._user_iou_thresholds = iou_thresholds or [x / 100.0 for x in range(50, 100, 5)]
        self.iou_thresholds = self._add_constant_weight(
            "iou_thresholds", self._user_iou_thresholds
        )
        # TODO(lukewood): support inference of category_ids based on update_state calls.
        self.category_ids = self._add_constant_weight("category_ids", category_ids)

        # default area ranges are defined for the COCO set
        # 32 ** 2 represents a 32x32 object.
        area_range = area_range or [0, 1e9**2]
        self.area_range = self._add_constant_weight(
            "area_range",
            area_range or [],
            shape=(2,),
            dtype=tf.float32,
        )
        self.max_detections = self._add_constant_weight(
            "max_detections", max_detections or 100, dtype=tf.int32
        )

        # Initialize result counters
        t = self.iou_thresholds.shape[0]
        k = self.category_ids.shape[0]
        a = self.area_ranges.shape[0]
        m = self.max_detections.shape[0]

        self.true_positives = self.add_weight(
            name="true_positives",
            shape=(t, k),
            dtype=tf.float32,
            initializer=initializers.Zeros(),
        )
        self.false_positives = self.add_weight(
            name="false_positives",
            shape=(t, k),
            dtype=tf.float32,
            initializer=initializers.Zeros(),
        )
        self.ground_truth_boxes = self.add_weight(
            name="ground_truth_boxes",
            shape=(k,),
            dtype=tf.float32,
            initializer=initializers.Zeros(),
        )

    def reset_state(self):
        super(COCOBase, self).reset_state()
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))
        self.ground_truth_boxes.assign(tf.zeros_like(self.ground_truth_boxes))

    @tf.function(jit_compile=True)
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Args:
            y_true: a bounding box Tensor in corners format.
            y_pred: a bounding box Tensor in corners format.
            sample_weight: Currently unsupported.
        """
        if sample_weight:
            raise NotImplementedError(
                "sample_weight is not yet supported in keras_cv COCO metrics."
            )
        num_images = y_true.shape[0]

        k = self.category_ids.shape[0]
        t = self.iou_thresholds.shape[0]

        # Sort by bbox.CONFIDENCE to make maxDetections easy to compute.
        y_pred = util.sort_bboxes(y_pred, axis=bbox.CONFIDENCE)
        true_positives_update = tf.zeros_like(self.true_positives)
        false_positives_update = tf.zeros_like(self.false_positives)
        ground_truth_boxes_update = tf.zeros_like(self.ground_truth_boxes)
        

        for img in tf.range(num_images):
            sentinel_filtered_y_true = util.filter_out_sentinels(y_true[img])
            sentinel_filtered_y_pred = util.filter_out_sentinels(y_pred[img])

            # Filter boxes by area
            # filter boxes by category

            for k_i in tf.range(k):
                category = self.category_ids[k_i]
                category_filtered_y_true = util.filter_boxes(
                    sentinel_filtered_y_true, value=category, axis=bbox.CLASS
                )
                category_filtered_y_pred = util.filter_boxes(
                    sentinel_filtered_y_pred, value=category, axis=bbox.CLASS
                )

                for a_i in tf.range(a):
                    area_range = self.area_ranges[a_i]
                    min_area = area_range[0]
                    max_area = area_range[1]
                    area_filtered_y_true = util.filter_boxes_by_area_range(
                        category_filtered_y_true, min_area, max_area
                    )
                    area_filtered_y_pred = category_filtered_y_pred  # area_filtered_y_pred = util.filter_boxes_by_area_range(category_filtered_y_pred, min_area, max_area)
                    ious = iou_lib.compute_ious_for_image(
                        area_filtered_y_true, area_filtered_y_pred
                    )

                    ground_truth_boxes_update = tf.tensor_scatter_nd_add(
                        ground_truth_boxes_update,
                        [[k_i, a_i]],
                        [tf.cast(tf.shape(area_filtered_y_true)[0], tf.float32)],
                    )

                    for t_i in tf.range(t):
                        threshold = self.iou_thresholds[t_i]
                        gt_matches, pred_matches = self._match_boxes(
                            area_filtered_y_true, area_filtered_y_pred, threshold, ious
                        )
                        true_positives = tf.cast(pred_matches != -1, tf.float32)
                        false_positives = tf.cast(pred_matches == -1, tf.float32)

                        for m_i in range(m):
                            max_dets = self.max_detections[m_i]
                            indices = [t_i, k_i, a_i, m_i]
                            mdt_slice = tf.math.minimum(
                                tf.shape(false_positives)[0], max_dets
                            )
                            false_positives_sum = tf.math.reduce_sum(
                                false_positives[:mdt_slice], axis=-1
                            )
                            true_positives_sum = tf.math.reduce_sum(
                                true_positives, axis=-1
                            )

                            true_positives_update = tf.tensor_scatter_nd_add(
                                true_positives_update, [indices], [true_positives_sum],
                            )
                            false_positives_update = tf.tensor_scatter_nd_add(
                                false_positives_update,
                                [indices],
                                [false_positives_sum],
                            )

        self.true_positives.assign_add(true_positives_update)
        self.false_positives.assign_add(false_positives_update)
        self.ground_truth_boxes.assign_add(ground_truth_boxes_update)

    def _match_boxes(self, y_true, y_pred, threshold, ious):
        n_true = tf.shape(y_true)[0]
        n_pred = tf.shape(y_pred)[0]

        gt_matches = tf.TensorArray(
            tf.int32,
            size=n_true,
            dynamic_size=False,
            infer_shape=False,
            element_shape=(),
        )
        pred_matches = tf.TensorArray(
            tf.int32,
            size=n_pred,
            dynamic_size=False,
            infer_shape=False,
            element_shape=(),
        )
        for i in tf.range(n_true):
            gt_matches = gt_matches.write(i, -1)
        for i in tf.range(n_pred):
            pred_matches = pred_matches.write(i, -1)

        for detection_idx in tf.range(n_pred):
            m = -1
            iou = tf.math.minimum(threshold, 1 - 1e-10)

            for gt_idx in tf.range(n_true):
                if gt_matches.gather([gt_idx]) > -1:
                    continue
                # TODO(lukewood): update clause to account for gtIg
                # if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:

                if not ious[gt_idx, detection_idx] >= threshold:
                    continue
                iou = ious[gt_idx, detection_idx]
                m = gt_idx

            # Write back the match indices
            pred_matches = pred_matches.write(detection_idx, m)
            if m == -1:
                continue
            gt_matches = gt_matches.write(m, detection_idx)
        return gt_matches.stack(), pred_matches.stack()

    def result(self):
        raise NotImplementedError("COCOBase subclasses must implement `result()`.")

    def _add_constant_weight(self, name, values, shape=None, dtype=tf.float32):
        shape = shape or (len(values),)
        return self.add_weight(
            name=name,
            shape=shape,
            initializer=initializers.Constant(tf.cast(tf.constant(values), dtype)),
            dtype=dtype,
        )
