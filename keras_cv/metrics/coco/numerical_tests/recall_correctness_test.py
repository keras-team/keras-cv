"""Tests to ensure that COCOrecall computes the correct values.."""
import os

import numpy as np
import tensorflow as tf

from keras_cv.metrics.coco import iou as iou_lib
from keras_cv.metrics.coco.recall import COCORecall

SAMPLE_FILE = os.path.dirname(os.path.abspath(__file__)) + "/sample_boxes.npz"


"""
cocoeval.py outputs:

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.661
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.793
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.651
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.676
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.671
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.504
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.686
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.686
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.674
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.681
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.682
"""


class RecallCorrectnesstTest(tf.test.TestCase):
    def test_recall_correctness_maxdets_1(self):
        npzfile = np.load(SAMPLE_FILE)
        y_true = npzfile["arr_0"].astype(np.float32)
        y_pred = npzfile["arr_1"].astype(np.float32)

        categories = set(int(x) for x in y_true[:, :, 4].flatten())
        categories = [x for x in categories if x != -1]

        # Area range all
        recall = COCORecall(
            category_ids=categories, max_detections=[1], area_ranges=[(0, 1e5 ** 2)]
        )

        recall.update_state(y_true, y_pred)
        recall = recall.result().numpy()

        self.assertAlmostEqual(recall, 0.504)
        # recall_metric = #
