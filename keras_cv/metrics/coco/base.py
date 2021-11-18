from keras_cv import bbox
from keras_cv.metrics.coco import iou as iou_lib


class COCOBase(tf.keras.metrics.Metric):
    """COCOBase serves as a base for COCORecall and COCOPrecision.

    Args:
        iou_thresholds: defaults to [0.5:0.05:0.95].  Dimension T=len(iou_thresholds), defaults to 10.
        category_ids: no default, users must provide.  K=len(category_ids)
        recall_thresholds: recall thresholds over which to compute  precision values, R=len(recall_thresholds).
        area_ranges: ranges to consider detections in

    Internally the COCOBase class tracks the following values:
    - precision: tf.Tensor with shape [TxRxKxAxM] precision for every evaluation setting.
    - recall: tf.Tensor with shape [TxKxAxM] max recall for every evaluation setting.
    """

    def __init__(
        self,
        iou_thresholds=None,
        category_ids=None,
        recall_thresholds=None,
        area_ranges=None,
        max_detections=None,
    ):
        # Initialize parameter values
        self.iou_thresholds = self._add_constant_weight(
            "iou_thresholds", iou_thresholds or [x / 100.0 for x in range(50, 100, 5)]
        )
        # TODO(lukewood): support inference of category_ids based on update_state calls.
        self.category_ids = self._add_constant_weight("category_ids", category_ids)
        self.recall_thresholds = self._add_constant_weight(
            "recall_thresholds",
            recall_thresholds or [x / 100.0 for x in range(0, 1, 0.01)],
        )

        """
        [[0 ** 2, 1e5 ** 2],   # all
        [0 ** 2, 32 ** 2],      # small 
        [32 ** 2, 96 ** 2],    # medium
        [96 ** 2, 1e5 ** 2]]  #  large
        """
        self.area_ranges = self._add_constant_weight(
            "area_ranges", area_ranges or [], shape=(len(area_ranges), 2)
        )
        self.max_detections = self._add_constant_weight(
            "max_detections", max_detections or [1, 10, 100]
        )

        # Initialize result counters
        k = self.category_ids.shape[0]
        t = self.iou_thresholds.shape[0]
        r = self.recall_thresholds.shape[0]
        a = self.area_ranges.shape[0]
        m = self.max_detections.shape[0]

        self.precision = self.add_weight(
            name="precision",
            shape=(t, r, k, a, m),
            trainable=False,
            dtype=tf.float32,
            initializer=initializers.Zeros(),
        )
        self.recall = self.add_weight(
            name="recall",
            shape=(t, k, a, m),
            trainable=False,
            dtype=tf.float32,
            initializer=initializers.Zeros(),
        )

    def _add_constant_weight(self, name, values, shape=None):
        shape = shape or (len(values),)
        return self.add_weight(
            name=name,
            shape=shape,
            initializer=initializers.Constant(values),
            dtype=tf.float32,
            trainable=False,
        )
