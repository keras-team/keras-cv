import sys

import tensorflow as tf
from absl import flags
from absl import logging

import keras_cv.layers.preprocessing
from keras_cv.datasets.pascal_voc.segmentation import load

flags.DEFINE_string(
    "model_name", None, "The name of the model in KerasCV.models to use."
)
flags.DEFINE_string("imagenet_path", None, "Directory from which to load Imagenet.")
flags.DEFINE_string(
    "backup_path", None, "Directory which will be used for training backups."
)
flags.DEFINE_string(
    "weights_path", None, "Directory which will be used to store weight checkpoints."
)
flags.DEFINE_string(
    "tensorboard_path", None, "Directory which will be used to store tensorboard logs."
)
flags.DEFINE_integer(
    "batch_size",
    4,
    "Batch size for training and evaluation. This will be multiplied by the number of accelerators in use.",
)
flags.DEFINE_boolean(
    "use_xla", True, "Whether or not to use XLA (jit_compile) for training."
)
flags.DEFINE_boolean(
    "use_mixed_precision",
    False,
    "Whether or not to use FP16 mixed precision for training.",
)
flags.DEFINE_boolean(
    "use_ema",
    True,
    "Whether or not to use exponential moving average weight updating",
)
flags.DEFINE_boolean(
    "include_rescaling",
    True,
    "Whether or not to include rescaling in the model",
)
flags.DEFINE_string(
    "backbone_weights",
    "imagenet",
    "Path for the weights to be loaded into the backbone",
)
flags.DEFINE_string(
    "backbone",
    None,
    "String denoting a supported backbone for the segmentation model",
)
flags.DEFINE_float(
    "initial_learning_rate",
    0.007,
    "Initial learning rate which will reduce on plateau. This will be multiplied by the number of accelerators in use",
)
flags.DEFINE_string(
    "model_kwargs",
    "{}",
    "Keyword argument dictionary to pass to the constructor of the model being trained",
)

flags.DEFINE_float(
    "weight_decay",
    5e-4,
    "Weight decay parameter for the optimizer",
)

# An upper bound for number of epochs (this script uses EarlyStopping).
flags.DEFINE_integer("epochs", 1000, "Epochs to train for")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

# Try to detect an available TPU. If none is present, default to MirroredStrategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
    if FLAGS.mixed_precision:
        logging.info("Mixed precision training enabled")
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
except ValueError:
    # MirroredStrategy is best for a single machine with one or multiple GPUs
    strategy = tf.distribute.MirroredStrategy()
    if FLAGS.mixed_precision:
        logging.info("Mixed precision training enabled")
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
print("Number of accelerators: ", strategy.num_replicas_in_sync)

local_batch = FLAGS.batch_size
global_batch = local_batch * strategy.num_replicas_in_sync
base_lr = FLAGS.initial_learning_rate * strategy.num_replicas_in_sync

train_ds = load(split="sbd_train", data_dir=None)
eval_ds = load(split="sbd_eval", data_dir=None)


def preprocess_image(img, cls_seg):
    img = tf.keras.layers.Resizing(512, 512, interpolation="nearest")(img)
    img, cls_seg = keras_cv.layers.preprocessing.RandomFlip("horizontal")(img, cls_seg)
    return img, cls_seg


def proc_train_fn(examples):
    image = examples.pop("image")
    cls_seg = examples.pop("class_segmentation")

    image, cls_seg = preprocess_image(image, cls_seg)

    sample_weight = tf.equal(cls_seg, 255)
    zeros = tf.zeros_like(cls_seg)
    cls_seg = tf.where(sample_weight, zeros, cls_seg)
    return image, cls_seg


def proc_eval_fn(examples):
    image = examples.pop("image")
    image = tf.cast(image, tf.float32)
    image = tf.keras.layers.Resizing(512, 512, interpolation="nearest")(image)
    cls_seg = examples.pop("class_segmentation")
    cls_seg = tf.cast(cls_seg, tf.float32)
    cls_seg = tf.keras.layers.Resizing(512, 512, interpolation="nearest")(cls_seg)
    cls_seg = tf.cast(cls_seg, tf.uint8)
    sample_weight = tf.equal(cls_seg, 255)
    zeros = tf.zeros_like(cls_seg)
    cls_seg = tf.where(sample_weight, zeros, cls_seg)
    return image, cls_seg


train_ds = train_ds.map(proc_train_fn, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(global_batch, drop_remainder=True)

eval_ds = eval_ds.map(proc_eval_fn, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.batch(global_batch, drop_remainder=True)

train_ds = train_ds.shuffle(8)
train_ds = train_ds.prefetch(2)


with strategy.scope():
    lr_decay = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[30000 * 16 / global_batch],
        values=[base_lr, 0.1 * base_lr],
    )
    model = keras_cv.models.__dict__[FLAGS.model_name]
    model = model(
        classes=21,
        backbone=FLAGS.backbone,
        include_rescaling=FLAGS.include_rescaling,
        backbone_weights=FLAGS.backbone_weights,
        **eval(FLAGS.model_kwargs)
    )
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_decay,
        momentum=0.9,
        clipnorm=10.0,
        use_ema=FLAGS.use_ema,
        weight_decay=FLAGS.weight_decay,
    )
    # ignore 255 as the class for semantic boundary.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(ignore_class=255)
    metrics = [
        tf.keras.metrics.SparseCategoricalCrossentropy(ignore_class=255),
        tf.keras.metrics.MeanIoU(num_classes=21, sparse_y_pred=False),
        tf.keras.metrics.SparseCategoricalAccuracy(),
    ]

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=FLAGS.weights_path,
        monitor="val_mean_io_u",
        save_best_only=True,
        save_weights_only=True,
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir=FLAGS.tensorboard_path, write_steps_per_second=True
    ),
]
model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

model.fit(train_ds, epochs=100, validation_data=eval_ds, callbacks=callbacks)
