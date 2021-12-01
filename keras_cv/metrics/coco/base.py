from keras_cv import bbox
from keras_cv.metrics.coco import iou as iou_lib
from keras_cv.metrics.coco import util


class COCOBase(tf.keras.metrics.Metric):
    """COCOBase serves as a base for COCORecall and COCOPrecision.

    Args:
        iou_thresholds: defaults to [0.5:0.05:0.95].  Dimension T=len(iou_thresholds), defaults to 10.
        category_ids: no default, users must provide.  K=len(category_ids)
        recall_thresholds: recall thresholds over which to compute  precision values, R=len(recall_thresholds).
        area_ranges: ranges to consider detections in, defaults to [all, 0-32, 32-96, 96>].

    Internally the COCOBase class tracks the following values:
    - precision: tf.Tensor with shape [TxRxKxAxM] precision for every evaluation setting.
    - recall: tf.Tensor with shape [TxKxAxM] max recall for every evaluation setting.
    """

    def __init__(
        self,
        iou_thresholds=None,
        category_ids=None,
        area_ranges=None,
        max_detections=None,
    ):
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
            "area_ranges", area_ranges or [], shape=(len(area_ranges), 2)
        )
        self.max_detections = self._add_constant_weight(
            "max_detections", max_detections or [1, 10, 100]
        )

        # Initialize result counters
        t = self.iou_thresholds.shape[0]
        k = self.category_ids.shape[0]
        a = self.area_ranges.shape[0]
        m = self.max_detections.shape[0]

        self.true_positives = self.add_weight(
            name="true_positives",
            shape=(t, k, a, m),
            trainable=False,
            dtype=tf.float32,
            initializer=initializers.Zeros(),
        )
        self.false_positives = self.add_weight(
            name="false_positives",
            shape=(t, k, a, m),
            trainable=False,
            dtype=tf.float32,
            initializer=initializers.Zeros(),
        )
        self.ground_truth_boxes = self.add_weight(
            name="ground_truth_boxes",
            shape=(k, a, m),
            trainable=False,
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

        ious = tf.TensorArray(
            tf.float32, size=num_images, dynamic_size=False, infer_shape=False
        )
        for img in tf.range(num_images):
            # iou lookup table per category.
            img_ious = tf.TensorArray(
                tf.float32, size=k, dynamic_size=False, infer_shape=False
            )
            for category_idx in tf.range(k):
                category = self.category_ids[category_idx]
                # filter_boxes automatically filters out categories set to -1
                # this includes our sentinel boxes padded out.
                filtered_y_true = util.filter_boxes(
                    y_true[img], value=category, axis=bbox.CLASS
                )
                filtered_y_pred = util.filter_boxes(
                    y_pred[img], value=category, axis=bbox.CLASS
                )
                per_category_ious = iou_lib.compute_ious_for_image(
                    filtered_y_true, filtered_y_pred
                )

                for iou_idx in tf.range(t):
                    iou_threshold = self.iou_thresholds[iou_idx]

                    # detect matches, detect ignores, etc

                    for max_dts_idx in tf.range(m):
                        max_dts = self.max_detections[max_dts_idx]

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

    def _add_constant_weight(self, name, values, shape=None):
        shape = shape or (len(values),)
        return self.add_weight(
            name=name,
            shape=shape,
            initializer=initializers.Constant(values),
            dtype=tf.float32,
            trainable=False,
        )
