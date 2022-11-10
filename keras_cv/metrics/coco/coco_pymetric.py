import six
import tensorflow as tf
from keras.callbacks import Callback

from keras_cv import bounding_box
from keras_cv.metrics.coco.pycoco_utils import compute_pycoco_metrics


class COCOPyMetric(Callback):
    def __init__(self, bounding_box_format):
        self.model = None
        self.bounding_box_format = bounding_box_format
        super().__init__()

    def set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        dataset = self.model._eval_data_handler._dataset
        gt = {}
        preds = {}

        for i, batch in enumerate(dataset):
            gt_i, preds_i = self._eval_batch(batch, i)

            for k, v in six.iteritems(preds_i):
                if k not in preds:
                    preds[k] = [v]
                else:
                    preds[k].append(v)

            for k, v in six.iteritems(gt_i):
                if k not in gt:
                    gt[k] = [v]
                else:
                    gt[k].append(v)

        metrics = compute_pycoco_metrics(gt, preds)

        logs.update(metrics)

    def _eval_batch(self, batch, index):
        images, gt_boxes = batch
        batch_size = images.shape[0]
        height = images.shape[1]
        width = images.shape[2]

        gt_boxes = bounding_box.convert_format(
            gt_boxes, source=self.bounding_box_format, target="yxyx"
        )

        gt_classes = gt_boxes[:, :, 4]
        gt_boxes = gt_boxes[:, :, :4]

        source_ids = tf.strings.join(
            [
                tf.strings.as_string(tf.tile(tf.constant([index]), [batch_size])),
                tf.strings.as_string(
                    tf.linspace(1, batch_size, batch_size), precision=0
                ),
            ],
            separator="/",
        )

        ground_truth = {}
        ground_truth["source_id"] = source_ids
        ground_truth["height"] = tf.tile(tf.constant([height]), [batch_size])
        ground_truth["width"] = tf.tile(tf.constant([width]), [batch_size])

        num_dets = gt_classes.get_shape().as_list()[1]
        ground_truth["num_detections"] = tf.tile(tf.constant([num_dets]), [batch_size])
        ground_truth["boxes"] = gt_boxes
        ground_truth["classes"] = gt_classes

        y_pred = self.model.predict(images)
        y_pred = bounding_box.convert_format(
            y_pred, source=self.bounding_box_format, target="yxyx"
        )

        predictions = {}
        predictions["num_detections"] = y_pred.row_lengths()
        y_pred = y_pred.to_tensor(-1)

        predictions["source_id"] = source_ids
        predictions["detection_boxes"] = y_pred[:, :, :4]
        predictions["detection_classes"] = y_pred[:, :, 4]
        predictions["detection_scores"] = y_pred[:, :, 5]

        return ground_truth, predictions
