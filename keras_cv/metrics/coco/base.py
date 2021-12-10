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
        recall_thresholds: recall thresholds over which to compute  precision values, R=len(recall_thresholds).
        area_ranges: ranges to consider detections in, defaults to [all, 0-32, 32-96, 96>].

    Internally the COCOBase class tracks the following values:
    - TruePositives: tf.Tensor with shape [TxKxAxM] precision for every evaluation setting.
    - FalsePositives: tf.Tensor with shape [TxKxAxM] precision for every evaluation setting.
    - GroundTruthBoxes: tf.Tensor with shape [KxAxM] max recall for every evaluation setting.
    """

    def __init__(
        self,
        iou_thresholds=None,
        category_ids=None,
        area_ranges=None,
        max_detections=None,
        **kwargs
    ):
        super(COCOBase, self).__init__(**kwargs)
        # Initialize parameter values
        self.iou_thresholds = self._add_constant_weight(
            "iou_thresholds", iou_thresholds or [x / 100.0 for x in range(50, 100, 5)]
        )
        # TODO(lukewood): support inference of category_ids based on update_state calls.
        self.category_ids = self._add_constant_weight("category_ids", category_ids)

        # default area ranges are defined for the COCO set
        # 32 ** 2 represents a 32x32 object.
        area_ranges = area_ranges or [
            [0 ** 2, 1e5 ** 2],  # all objects
            [0 ** 2, 32 ** 2],  # small objects
            [32 ** 2, 96 ** 2],  # medium size objects
            [96 ** 2, 1e5 ** 2],  # large size objects
        ]
        self.area_ranges = self._add_constant_weight(
            "area_ranges",
            area_ranges or [],
            shape=(len(area_ranges), 2),
            dtype=tf.int32,
        )
        self.max_detections = self._add_constant_weight(
            "max_detections", max_detections or [1, 10, 100], dtype=tf.int32
        )

        # Initialize result counters
        t = self.iou_thresholds.shape[0]
        k = self.category_ids.shape[0]
        a = self.area_ranges.shape[0]
        m = self.max_detections.shape[0]

        self.true_positives = self.add_weight(
            name="true_positives",
            shape=(t, k, a, m),
            dtype=tf.float32,
            initializer=initializers.Zeros(),
        )
        self.false_positives = self.add_weight(
            name="false_positives",
            shape=(t, k, a, m),
            dtype=tf.float32,
            initializer=initializers.Zeros(),
        )
        self.ground_truth_boxes = self.add_weight(
            name="ground_truth_boxes",
            shape=(k, a, m),
            dtype=tf.float32,
            initializer=initializers.Zeros(),
        )

    def reset_state(self):
        super(COCOBase, self).reset_state()
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))
        self.ground_truth_boxes.assign(tf.zeros_like(self.ground_truth_boxes))

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight:
            raise NotImplementedError(
                "sample_weight is not yet supported in keras_cv COCO metrics."
            )

        num_images = y_true.shape[0]

        k = self.category_ids.shape[0]
        t = self.iou_thresholds.shape[0]
        a = self.area_ranges.shape[0]
        m = self.max_detections.shape[0]

        # first, we prepare eval_imgs.  eval_imgs creates a lookup table from
        # [image_id, category_id, area_ranges] => results
        # this is equivalent to the step of `evaluate()` in cocoeval

        # evaluate first computes ious for all images in a dictionary.  The dictionary maps
        # from imgId, catId to iou scores.

        # in our implementation we will iterate imgId and catId and store the ious in a lookup
        # Tensor with the dimensions [image_id, category_id, bbox_true, bbox_pred] => iou

        # Sort by bbox.CONFIDENCE to make maxDetections easy to compute.
        y_pred = util.sort_bboxes(y_pred, axis=bbox.CONFIDENCE)

        for img in tf.range(num_images):
            # iou lookup table per category.
            img_ious = tf.TensorArray(
                tf.float32, size=k, dynamic_size=False, infer_shape=False
            )

            true_positives_update_result = tf.TensorArray(
                tf.float32, size=k, dynamic_size=False
            )
            false_positives_update_result = tf.TensorArray(
                tf.float32, size=k, dynamic_size=False
            )

            a_i = 0  # TODO(lukewood): use area ranges

            for category_idx in tf.range(k):
                category = self.category_ids[category_idx]
                # filter_boxes automatically filters out categories set to -1
                # this includes our sentinel boxes padded out.
                filtered_y_true = util.filter_boxes(
                    y_true[img], value=category, axis=bbox.CLASS
                )
                filtered_y_true = util.filter_out_sentinels(filtered_y_true)

                filtered_y_pred = util.filter_boxes(
                    y_pred[img], value=category, axis=bbox.CLASS
                )
                filtered_y_pred = util.filter_out_sentinels(filtered_y_pred)

                n_true = tf.shape(filtered_y_true)[0]
                n_pred = tf.shape(filtered_y_pred)[0]

                # TODO(lukewood): filter area ranges

                ious = iou_lib.compute_ious_for_image(filtered_y_true, filtered_y_pred)

                # TensorArray so we can regularly write back to the array
                gt_matches_outer = tf.TensorArray(
                    tf.int32,
                    size=t,
                    dynamic_size=False,
                )
                pred_matches_outer = tf.TensorArray(
                    tf.int32,
                    size=t,
                    dynamic_size=False,
                )
                
                tp_a_result = tf.TensorArray(tf.float32, size=a, dynamic_size=False)
                fp_a_result = tf.TensorArray(tf.float32, size=a, dynamic_size=False)
                
                # TODO(lukewood): account for area ranges
                for tind in range(t):
                    threshold = self.iou_thresholds[tind]

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

                    for detection_idx in tf.range(n_pred):

                        # initialize the match index to -1
                        # "iou to beat" is set to threshold
                        m = -1
                        iou = tf.math.minimum(threshold, 1 - 1e-10)

                        for gt_idx in tf.range(n_true):
                            if gt_matches.gather([gt_idx]) > 0:
                                continue
                            # TODO(lukewood): update clause to account for gtIg
                            # if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:

                            if not ious[detection_idx, gt_idx] >= threshold:
                                continue
                            iou = ious[detection_idx, gt_idx]
                            m = gt_idx

                        # Write back the match indices
                        pred_matches = pred_matches.write(detection_idx, m)
                        if m == -1:
                            continue
                        gt_matches = gt_matches.write(m, detection_idx)

                    gt_matches_outer = gt_matches_outer.write(tind, gt_matches.stack())
                    pred_matches_outer = pred_matches_outer.write(
                        tind, pred_matches.stack()
                    )
                pred_matches = pred_matches_outer.stack()
                gt_matches = gt_matches_outer.stack()

                true_positives = tf.cast(pred_matches != -1, tf.float32)
                false_positives = tf.cast(pred_matches == -1, tf.float32)

                m = tf.shape(self.max_detections)[0]

                m_true_positives_result = tf.TensorArray(
                    tf.float32, size=m, dynamic_size=False
                )
                m_false_positives_result = tf.TensorArray(
                    tf.float32, size=m, dynamic_size=False
                )
                for m_i in tf.range(m):
                    max_dets = self.max_detections[m_i]
                    # TODO(lukewood): cross check maxdets logic against cocoeval.py
                    # TODO(lukewood): use max dets
                    false_positives_sum = tf.math.reduce_sum(false_positives, axis=-1)
                    true_positives_sum = tf.math.reduce_sum(true_positives, axis=-1)

                    m_true_positives_result = m_true_positives_result.write(
                        m_i, true_positives_sum
                    )
                    m_false_positives_result = m_false_positives_result.write(
                        m_i, false_positives_sum
                    )
                
                m_true_positives_result = tf.transpose(m_true_positives_result.stack(), perm=[1, 0])
                m_false_positives_result = tf.transpose(m_false_positives_result.stack(), perm=[1, 0])

                tp_a_result = tp_a_result.write(a_i, m_true_positives_result)
                fp_a_result = fp_a_result.write(a_i, m_true_positives_result)

                tp_a_result = tf.transpose(tp_a_result.stack(), perm=[1, 0, 2])
                fp_a_result = tf.transpose(fp_a_result.stack(), perm=[1, 0, 2])

                true_positives_update_result = true_positives_update_result.write(
                    category_idx, tp_a_result
                )
                false_positives_update_result = false_positives_update_result.write(
                    category_idx, fp_a_result
                )

            tp_update = true_positives_update_result.stack()
            fp_update = false_positives_update_result.stack()

            tp_update = tf.transpose(tp_update, perm=[1, 0, 2, 3])
            fp_update = tf.transpose(fp_update, perm=[1, 0, 2, 3])
            
            self.true_positives.assign_add(tp_update)
            self.false_positives.assign_add(fp_update)

            # tp_update = tf.transpose(tp_update, perm=[])
            # fp_update = tf.transpose(fp_update, perm=[])
        # next, for each image we compute:
        # - dtIgnore: [imgId, catId, areaRange] => mask
        # - gtIgnore: [imgId, catId, areaRange] => mask
        # - dtMatches: [catId, areaRange] =>

        # - dtScores is already stored in y_pred[imgId, bbox.CONFIDENCE]

        # the next section is equivalent to the `accumulate()` step in cocoeval

        # in the original implementation they fetch all of the values in evalImgs using:
        # [image_id, category_id, area_range]
        # then, we stack dtScores[0:maxDet].  This is equivalent to y_pred[:, 0:maxdet, 5]
        # in our implementation our dtScores will have -1s for sentinel missing values.

        # next, they craft an indices ordering set using np.argsort(-dtscores, axis=-1).
        # this axis set is used to sort:
        # y_pred[dtMatches], y_pred[dtIgnore].  We will create dtignore the same way,
        # with the additional mask out of our padded -1 values bboxes.

        # next we check if gtIgnore, which is stored in a Tensor computed by the evaluate()
        # section, has any non_zero values for the current image/catid/area_range.
        # if gtIf is all zeros we just continue in the loop

        # next, true positives is computed using a logical and of sorted dts, and a not of sorted ignores
        # false positives are computed using the inverse of the true positives, so logical_not(dts)

        # a sum is taken of true positives and false positives on axis=1.  This contins the result

        # now, a pretty complex loop takes place:
        # https://source.corp.google.com/piper///depot/google3/third_party/py/pycocotools/cocoeval.py;l=409
        # the summary is that it computes the recall and precision based on the tps, fps, etc.
        # result is stored in self.recall, and self.precision

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
