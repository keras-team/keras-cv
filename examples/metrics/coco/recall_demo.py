import tensorflow as tf
import keras_cv

NUM_CLASSES = 2

recall = keras_cv.metrics.COCORecall(
    max_detections=100,
    class_ids=range(NUM_CLASSES),
    area_range=(0, 64**2),
)

y_true = tf.ragged.stack(
    [
        tf.constant([[0, 0, 10, 10, 1], [5, 5, 10, 10, 0]], tf.float32),
        tf.constant([[0, 0, 10, 10, 1]], tf.float32),
    ]
)
y_pred = tf.ragged.stack(
    [
        tf.constant([[5, 5, 10, 10, 1, 0.9]], tf.float32),
        tf.constant([[0, 0, 10, 10, 1, 1.0], [5, 5, 10, 10, 0, 0.9]], tf.float32),
    ]
)

print("Recall for test set:", recall.update_state(y_true, y_pred))
