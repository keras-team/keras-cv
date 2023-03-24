"""
Title: Plot a bounding box gallery
Author: [lukewood](https://lukewood.xyz)
Date created: 2023/03/22
Last modified: 2023/03/22
Description: Visualize bounding boxes for a given dataset.
"""

"""
`keras_cv.visualization.plot_bounding_box_gallery()` is a function dedicated to the
visualization of bounding boxes predicted by a `keras_cv` object detection model.
"""

import tensorflow as tf
import tensorflow_datasets as tfds

import keras_cv

"""
First, we load a dataset:
"""

train_ds = tfds.load(
    "voc/2007", split="train", with_info=False, shuffle_files=True
)


def unpackage_tfds_inputs(inputs):
    image = inputs["image"]
    image = tf.cast(image, tf.float32)
    boxes = inputs["objects"]["bbox"]
    boxes = keras_cv.bounding_box.convert_format(
        boxes,
        images=image,
        source="rel_yxyx",
        target="xywh",
    )
    classes = tf.cast(inputs["objects"]["label"], tf.float32)
    bounding_boxes = {"classes": classes, "boxes": boxes}
    return image, bounding_boxes


train_ds = train_ds.map(unpackage_tfds_inputs)
train_ds = train_ds.apply(tf.data.experimental.dense_to_ragged_batch(16))
images, boxes = next(iter(train_ds.take(1)))

"""
You can give the utility class IDs to annotate the drawn bounding boxes:
"""

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

"""
The function accepts `y_true`, `y_pred`, or both to visualize boxes:
"""
keras_cv.visualization.plot_bounding_box_gallery(
    images,
    value_range=(0, 255),
    bounding_box_format="xywh",
    y_true=boxes,
    scale=3,
    rows=2,
    cols=2,
    line_thickness=4,
    font_scale=1,
    legend=True,
    class_mapping=class_mapping,
)

"""
Same but with `y_pred`:
"""

keras_cv.visualization.plot_bounding_box_gallery(
    images,
    value_range=(0, 255),
    bounding_box_format="xywh",
    y_pred=boxes,
    scale=3,
    rows=2,
    cols=2,
    line_thickness=4,
    font_scale=1,
    legend=True,
    class_mapping=class_mapping,
)
