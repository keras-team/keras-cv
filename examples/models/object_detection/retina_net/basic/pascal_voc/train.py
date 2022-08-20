"""
Title: Train an Object Detection Model on Pascal VOC 2007 using KerasCV
Author: [lukewood](https://lukewood.xyz)
Date created: 2022/08/02
Last modified: 2022/08/02
Description: Use KerasCV to train a RetinaNet on Pascal VOC 2007.
"""

"""
## Overview

KerasCV offers a complete set of APIs to allow you to train your own state-of-the-art,
production-grade object detection model.  These APIs include object detection specific
data augmentation techniques, models, and COCO metrics.

To get started, let's sort out all of our imports and define global configuration parameters.
To support long training runs, experiment tracking via [Weights & Biases](https://wandb.ai),
and dynamic batch sizing, we define our configuration parameters as command line flags.
"""

import sys

import matplotlib.pyplot as plt
import tensorflow as tf
import wandb
from absl import flags
from tensorflow.keras import callbacks as callbacks_lib
from tensorflow.keras import optimizers

import keras_cv

flags.DEFINE_integer("batch_size", 8, "Training and eval batch size.")
flags.DEFINE_integer("epochs", 1, "Number of training epochs.")
flags.DEFINE_string("wandb_entity", None, "wandb entity to use.")
flags.DEFINE_string("experiment_name", None, "wandb run name to use.")
FLAGS = flags.FLAGS

FLAGS(sys.argv)

if FLAGS.wandb_entity:
    wandb.init(
        project="pascalvoc-retinanet",
        entity=FLAGS.wandb_entity,
        name=FLAGS.experiment_name,
    )

"""
## Data loading

In this guide, we use the data-loading function: `keras_cv.loaders.pascal_voc.load()`.
KerasCV supports a `bounding_box_format` argument in all components that process
bounding boxes.  To match the KerasCV API style, it is recommended that when writing a
custom data loader, you also support a `bounding_box_format` argument.
This makes it clear to those invoking your data loader what format the bounding boxes
are in.

For example:

```python
train_ds, ds_info = keras_cv.loaders.pascal_voc.load(split='train', bounding_box_format='xywh', batch_size=8)
```

Clearly yields bounding boxes in the format `xywh`.  You can read more about
KerasCV bounding box formats [in the API docs](https://keras.io/api/keras_cv/bounding_box/formats/).

Our data comesloaded into the format
`{"images": images, "bounding_boxes": bounding_boxes}`.  This format is supported in all
KerasCV preprocessing components.

Lets load some data and verify that our data looks as we expect it to.
"""

dataset, _ = keras_cv.loaders.pascal_voc.load(
    split="train", bounding_box_format="xywh", batch_size=9
)


def visualize_dataset(dataset, bounding_box_format):
    color = tf.constant(((255.0, 0, 0),))
    plt.figure(figsize=(10, 10))
    iterator = iter(dataset)
    for i in range(9):
        example = next(iterator)
        images, boxes = example["images"], example["bounding_boxes"]
        boxes = keras_cv.bounding_box.convert_format(
            boxes, source=bounding_box_format, target="rel_yxyx", images=images
        )
        boxes = boxes.to_tensor(default_value=-1)
        plotted_images = tf.image.draw_bounding_boxes(images, boxes[..., :4], color)
        plt.subplot(9 // 3, 9 // 3, i + 1)
        plt.imshow(plotted_images[0].numpy().astype("uint8"))
        plt.axis("off")
    plt.show()


visualize_dataset(dataset, bounding_box_format="xywh")

"""
Looks like everything is structured as expected.  Now we can move on to constructing our
data augmentation pipeline.
"""

"""
## Data augmentation

One of the most labor-intensive tasks when constructing object detection pipeliens is
data augmentation.  Image augmentation techniques must be aware of the underlying
bounding boxes, and must update them accordingly.

Luckily, KerasCV natively supports bounding box augmentation with its extensive library
of [data augmentation layers](https://keras.io/api/keras_cv/layers/preprocessing/).
The code below loads the Pascal VOC dataset, and performs on-the-fly bounding box
friendly data augmentation inside of a `tf.data` pipeline.
"""

# train_ds is batched as a (images, bounding_boxes) tuple
# bounding_boxes are ragged
train_ds, _ = keras_cv.loaders.pascal_voc.load(
    bounding_box_format="xywh", split="train", batch_size=FLAGS.batch_size
)
val_ds, _ = keras_cv.loaders.pascal_voc.load(
    bounding_box_format="xywh", split="validation", batch_size=FLAGS.batch_size
)

augmentation_layers = [
    # keras_cv.layers.RandomShear(x_factor=0.1, bounding_box_format='xywh'),
    # TODO(lukewood): add color jitter and others
]


def augment(sample):
    for layer in augmentation_layers:
        sample = layer(sample)
    return sample


train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
visualize_dataset(train_ds, bounding_box_format="xywh")

"""
Great!  We now have a bounding box friendly augmentation pipeline.

Next, let's unpackage our inputs from the preprocessing dictionary, and prepare to feed
the inputs into our model.
"""


def unpackage_dict(inputs):
    return inputs["images"], inputs["bounding_boxes"]


train_ds = train_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

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
    classes=20,
    bounding_box_format="xywh",
    backbone="resnet50",
    backbone_weights="imagenet",
    include_rescaling=True,
)

"""
That is all it takes to construct a KerasCV RetinaNet.  The RetinaNet accepts tuples of
dense image Tensors and ragged bounding box Tensors to `fit()` and `train_on_batch()`
This matches what we have constructed in our input pipeline above.

The RetinaNet `call()` method outputs two values: training targets and inference targets.
In this guide, we are primarily concerned with the inference targets.  Internally, the
training targets are used by `keras_cv.losses.ObjectDetectionLoss()` to train the
network.
"""

"""
## Optimizer

For training, we use a SGD optimizer with a piece-wise learning rate schedule
consisting of a warm up followed by a ramp up, then a ramp.
Below, we construct this using a `keras.optimizers.schedules.PiecewiseConstantDecay`
schedule.
"""

optimizer = optimizers.SGD(learning_rate=0.1, momentum=0.9, global_clipnorm=10.0)

"""
## COCO metrics monitoring

KerasCV offers a suite of in-graph COCO metrics that support batch-wise evaluation.
More information on these metrics is available in:

- [Efficient Graph-Friendly COCO Metric Computation for Train-Time Model Evaluation](https://arxiv.org/abs/2207.12120)
- [Using KerasCV COCO Metrics](https://keras.io/guides/keras_cv/coco_metrics/)

Lets construct two COCO metrics, an instance of
`keras_cv.metrics.COCOMeanAveragePrecision` with the parameterization to match the
standard COCO Mean Average Precision metric, and `keras_cv.metrics.COCORecall`
parameterized to match the standard COCO Recall metric.
"""

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

"""
## Training our model

All that is left to do is train our model.  KerasCV object detection models follow the
standard Keras workflow, leveraging `compile()` and `fit()`.

Let's compile our model:
"""

loss = keras_cv.losses.ObjectDetectionLoss(
    classes=20,
    classification_loss=keras_cv.losses.FocalLoss(from_logits=True, reduction="none"),
    box_loss=keras_cv.losses.SmoothL1Loss(l1_cutoff=1.0, reduction="none"),
    reduction="auto",
)

model.compile(
    loss=loss,
    optimizer=optimizer,
    metrics=metrics,
)

"""
All that is left to do is construct some callbacks:
"""

callbacks = [
    callbacks_lib.TensorBoard(log_dir="logs"),
    callbacks_lib.EarlyStopping(patience=50),
    callbacks_lib.ReduceLROnPlateau(patience=20),
]
if FLAGS.wandb_entity:
    callbacks += [
        wandb.keras.WandbCallback(save_model=False),
    ]

"""
And run `model.fit()`!
"""

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FLAGS.epochs,
    callbacks=callbacks,
)

"""
## Results and conclusions

KerasCV makes it easy to construct state-of-the-art object detection pipelines.  All of
the KerasCV object detection components can be used independently, but also have deep
integration with each other.  With KerasCV, bounding box augmentation, train-time COCO
metrics evaluation, and more, are all made simple and consistent.

By default, this script runs for a single epoch.  To run training to convergence,
invoke the script with a command line flag `--epochs=500`.  To save you the effort of
running the script for 500 epochs, we have produced a Weights and Biases report covering
the training results below!  As a bonus, the report includes a training run with and
without data augmentation.

[Metrics from a 500 epoch Weights and Biases Run are available here](
    https://tinyurl.com/y34xx65w
)
"""
