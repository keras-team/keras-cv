import numpy as np
import tensorflow as tf

from keras_cv import bounding_box

img = tf.zeros([20, 20, 3])
bboxes = np.array([[0, 0, 10, 10], [4, 4, 12, 12]])
bboxes = tf.convert_to_tensor(bboxes, dtype=tf.int32)
bounding_box.visualize_bounding_boxes_on_image(img, bboxes)
