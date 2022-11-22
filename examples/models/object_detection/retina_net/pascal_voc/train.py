import os
import resource

import tensorflow as tf
import tensorflow_datasets as tfds
from luketils import visualization
from tensorflow import keras

import keras_cv
from keras_cv import layers

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))
os.makedirs("artifacts/", exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 100
CHECKPOINT_PATH = "checkpoint/"

class_ids = [
    "Aeroplane",
    "Bicycle",
    "Bird",
    "Boat",
    "Bottle",
    "Bus",
    "Car",
    "Cat",
    "Chair",
    "Cow",
    "Dining Table",
    "Dog",
    "Horse",
    "Motorbike",
    "Person",
    "Potted Plant",
    "Sheep",
    "Sofa",
    "Train",
    "Tvmonitor",
    "Total",
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))


def unpackage_raw_tfds_inputs(inputs, bounding_box_format="xywh"):
    image = inputs["image"]
    image = tf.cast(image, tf.float32)
    gt_boxes = inputs["objects"]["bbox"]
    gt_boxes = keras_cv.bounding_box.convert_format(
        gt_boxes,
        images=image,
        source="rel_yxyx",
        target=bounding_box_format,
    )
    gt_classes = tf.cast(inputs["objects"]["label"], tf.float32)
    gt_classes = tf.expand_dims(gt_classes, axis=-1)
    return {
        "images": image,
        "bounding_boxes": tf.concat([gt_boxes, gt_classes], axis=-1),
    }


train_ds = tfds.load("voc/2007", split="train", with_info=False, shuffle_files=True)
train_ds = train_ds.concatenate(
    tfds.load("voc/2012", split="train+validation", with_info=False, shuffle_files=True)
)
eval_ds = tfds.load("voc/2007", split="test", with_info=False)

train_ds = train_ds.map(unpackage_raw_tfds_inputs, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.map(unpackage_raw_tfds_inputs, num_parallel_calls=tf.data.AUTOTUNE)

eval_resizing = layers.Resizing(
    640, 640, bounding_box_format="xywh", pad_to_aspect_ratio=True
)

augmenter = layers.Augmenter(
    [
        layers.RandomFlip(mode="horizontal", bounding_box_format="xywh"),
        layers.RandomAspectRatio(factor=(0.9, 1.1), bounding_box_format="xywh"),
        layers.JitteredResize(
            target_size=(640, 640),
            scale_factor=(0.8, 1.25),
            bounding_box_format="xywh",
        ),
    ]
)

train_ds = train_ds.apply(
    tf.data.experimental.dense_to_ragged_batch(BATCH_SIZE, drop_remainder=True)
)
train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.apply(
    tf.data.experimental.dense_to_ragged_batch(BATCH_SIZE, drop_remainder=True)
)
eval_ds = eval_ds.map(
    eval_resizing,
    num_parallel_calls=tf.data.AUTOTUNE,
)

"""
Looks like everything is structured as expected.  Now we can move on to constructing our
data augmentation pipeline.
"""
train_ds = train_ds.prefetch(2)
train_ds = train_ds.shuffle(BATCH_SIZE)
eval_ds = eval_ds.prefetch(2)

# visualize each dataset
visualization.plot_bounding_box_gallery(
    train_ds,
    value_range=(0, 255),
    rows=4,
    cols=4,
    scale=2,
    bounding_box_format="xywh",
    path="artifacts/train.png",
)
visualization.plot_bounding_box_gallery(
    eval_ds,
    value_range=(0, 255),
    rows=4,
    cols=4,
    scale=2,
    bounding_box_format="xywh",
    path="artifacts/eval.png",
)


def unpackage_dict(inputs):
    return inputs["images"], inputs["bounding_boxes"]


train_ds = train_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)

"""
Our data pipeline is now complete.  We can now move on to model creation and training.
"""

"""
## Model creation

We'll use the KerasCV API to construct a RetinaNet model.  In this tutorial we use
a pretrained ResNet50 backbone using weights.  In order to perform fine-tuning, we
freeze the backbone before training.  When `include_rescaling=True` is set, inputs to
the model are expected to be in the range `[0, 255]`.
"""

model = keras_cv.models.RetinaNet(
    # number of classes to be used in box classification
    classes=len(class_ids),
    # For more info on supported bounding box formats, visit
    # https://keras.io/api/keras_cv/bounding_box/
    bounding_box_format="xywh",
    # KerasCV offers a set of pre-configured backbones
    backbone="resnet50",
    # Each backbone comes with multiple pre-trained weights
    # These weights match the weights available in the `keras_cv.model` class.
    backbone_weights="imagenet",
    # include_rescaling tells the model whether your input images are in the default
    # pixel range (0, 255) or if you have already rescaled your inputs to the range
    # (0, 1).  In our case, we feed our model images with inputs in the range (0, 255).
    include_rescaling=True,
    # Typically, you'll want to set this to False when training a real model.
    # evaluate_train_time_metrics=True makes `train_step()` incompatible with TPU,
    # and also causes a massive performance hit.  It can, however be useful to produce
    # train time metrics when debugging your model training pipeline.
    evaluate_train_time_metrics=False,
)
# Fine-tuning a RetinaNet is as simple as setting backbone.trainable = False
model.backbone.trainable = False

metrics = [
    keras_cv.metrics.COCOMeanAveragePrecision(
        class_ids=range(20),
        bounding_box_format="xywh",
        name="Mean Average Precision",
    ),
    keras_cv.metrics.COCORecall(
        class_ids=range(20),
        bounding_box_format="xywh",
        max_detections=100,
        name="Recall",
    ),
]
optimizer = tf.optimizers.SGD(global_clipnorm=10.0)
model.compile(
    classification_loss="focal",
    box_loss="smoothl1",
    optimizer=optimizer,
    metrics=metrics,
)

callbacks = [
    keras.callbacks.TensorBoard(log_dir="logs"),
    keras.callbacks.ReduceLROnPlateau(patience=5),
    keras.callbacks.EarlyStopping(patience=10),
    keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_weights_only=True),
]

history = model.fit(
    train_ds,
    validation_data=eval_ds.take(10),
    epochs=100,
    callbacks=callbacks,
)

# take() to prevent numerical issues that arise when evaluating on many inputs
keras_cv_metrics = model.evaluate(eval_ds.take(20), return_dict=True)
print("Metrics:", keras_cv_metrics)
