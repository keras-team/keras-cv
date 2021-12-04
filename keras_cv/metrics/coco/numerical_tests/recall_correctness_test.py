"""Tests to ensure that COCOrecall computes the correct values.."""
import os

import numpy as np
import tensorflow as tf

from keras_cv.metrics.coco import iou as iou_lib

SAMPLE_FILE = os.path.dirname(os.path.abspath(__file__)) + '/sample_boxes.npz'

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
    def test_recall_correctness(self):
        npzfile = np.load(SAMPLE_FILE)
        y_true = npzfile['arr_0']
        y_pred = npzfile['arr_1']

        # recall_metric = #
