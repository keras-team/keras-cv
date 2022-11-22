import resource

import luketils
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

import keras_cv
from keras_cv import layers

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

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

train_ds = tfds.load("voc/2007", split="train", with_info=False, shuffle_files=False)


def unpackage_inputs(bounding_box_format):
    def apply(inputs):
        image = inputs["image"]
        image = tf.cast(image, tf.float32)
        gt_boxes = tf.cast(inputs["objects"]["bbox"], tf.float32)
        gt_classes = tf.cast(inputs["objects"]["label"], tf.float32)
        gt_classes = tf.expand_dims(gt_classes, axis=1)
        gt_boxes = keras_cv.bounding_box.convert_format(
            gt_boxes,
            images=image,
            source="rel_yxyx",
            target=bounding_box_format,
        )
        bounding_boxes = tf.concat([gt_boxes, gt_classes], axis=-1)
        return {"images": image, "bounding_boxes": bounding_boxes}

    return apply


train_ds = train_ds.map(unpackage_inputs(bounding_box_format="xywh"))
train_ds = train_ds.apply(
    tf.data.experimental.dense_to_ragged_batch(2, drop_remainder=True)
)

augmenter = layers.Augmenter(
    [
        layers.RandomFlip(mode="horizontal", bounding_box_format="xywh"),
        layers.RandomScale(factor=(0.75, 1.25), bounding_box_format="xywh"),
        layers.RandomRaggedCrop(
            height_factor=(0.7, 1.0),
            width_factor=(0.7, 1.0),
            bounding_box_format="xywh",
        ),
        layers.RandomAspectRatio(factor=(0.8, 1.2), bounding_box_format="xywh"),
        layers.Resizing(512, 512, bounding_box_format="xywh", pad_to_aspect_ratio=True),
    ]
)

train_ds = train_ds.map(augmenter)
inputs = next(iter(train_ds))

while input("q?") != "q":
    images, boxes = [], []
    for _ in range(8):
        inputs = next(iter(train_ds))
        x, y = inputs["images"], inputs["bounding_boxes"]
        images.append(x)
        boxes.append(y)
    x = tf.concat(images, axis=0)
    y = tf.concat(boxes, axis=0)
    luketils.visualization.plot_bounding_box_gallery(
        images=x,
        y_true=y,
        value_range=(0, 255),
        rows=4,
        cols=4,
        scale=3,
        bounding_box_format="xywh",
    )
