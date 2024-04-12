import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import keras

import keras_cv
from keras_cv.models import FasterRCNN

batch_size = 1
image_shape = (512, 512, 3)

images = keras.ops.ones((batch_size,) + image_shape)
labels = {
    "boxes": keras.ops.array(
        [
            [
                [0, 0, 100, 100],
                [100, 100, 200, 200],
                [300, 300, 100, 100],
            ]
        ],
        dtype="float32",
    ),
    "classes": keras.ops.array([[1, 1, 1]], dtype="float32"),
}

# Initialize the model
model = FasterRCNN(
    batch_size=batch_size,
    num_classes=2,
    bounding_box_format="xywh",
    backbone=keras_cv.models.ResNet50Backbone.from_preset(
        "resnet50_imagenet",
        input_shape=image_shape,
    ),
)

# Call the model
outputs = model(images)
print("outputs")
for key, value in outputs.items():
    print(f"{key}: {value.shape}")

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(),
    box_loss=keras.losses.Huber(),
    classification_loss=keras.losses.CategoricalCrossentropy(),
    rpn_box_loss=keras.losses.Huber(),
    rpn_classification_loss=keras.losses.BinaryCrossentropy(from_logits=True),
)

# Compute Loss from the model
loss = model.compute_loss(x=images, y=labels, y_pred=None, sample_weight=None)
print(loss)

# Train step
xs = keras.ops.ones((1, 512, 512, 3), "float32")
ys = {
    "classes": keras.ops.array([[1, 1, 1]], dtype="float32"),
    "boxes": keras.ops.array(
        [
            [
                [0, 0, 100, 100],
                [100, 100, 200, 200],
                [300, 300, 100, 100],
            ]
        ],
        dtype="float32",
    ),
}
import tensorflow as tf
ds = tf.data.Dataset.from_tensor_slices((xs, ys))
ds = ds.batch(1, drop_remainder=True)
model.fit(ds, epochs=1)