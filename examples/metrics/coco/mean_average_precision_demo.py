import numpy as np
import tensorflow as tf

from keras_cv.metrics import COCOMeanAveragePrecision

map = COCOMeanAveragePrecision(category_ids=[1])
# These would match if they were in the area range
y_true = np.array(
    [[[0, 0, 10, 10, 1], [0, 0, 10, 10, 2]], [[0, 0, 10, 10, 1], [0, 0, 10, 10, 2]]]
).astype(np.float32)
y_pred = np.array(
    [
        [
            [-100, 0, 10, 10, 1, 1.0],
            [0, 0, 10, 10, 1, 0.7],
        ],
        [
            [0, 0, 10, 10, 1, 0.5],
            [0, 0, 10, 10, 1, 0.5],
        ],
    ]
).astype(np.float32)

map.update_state(tf.constant(y_true), tf.constant(y_pred))

result = map.result()
print("Result", result)
