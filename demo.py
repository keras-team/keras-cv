import numpy as np

import keras_cv

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Note: We absolutely need this while creating the Backbone
batch_size = 32
image_shape = (512, 512, 3)

images = np.ones((batch_size,) + image_shape)
labels = {
    "boxes": np.array(
        [
            [
                [0, 0, 100, 100],
                [100, 100, 200, 200],
                [300, 300, 100, 100],
            ]
        ],
        dtype=np.float32,
    ),
    "classes": np.array([[1, 1, 1]], dtype=np.float32),
}
model = keras_cv.models.FasterRCNN(
    batch_size=batch_size,
    num_classes=20,
    bounding_box_format="xywh",
    backbone=keras_cv.models.ResNet50Backbone.from_preset(
        "resnet50_imagenet",
        input_shape=image_shape,
    ),
)

# # Evaluate model without box decoding and NMS
# model(images)

# # Prediction with box decoding and NMS
# model.predict(images)

# # Train model
# model.compile(
#     classification_loss="focal",
#     box_loss="smoothl1",
#     optimizer=keras.optimizers.SGD(global_clipnorm=10.0),
#     jit_compile=False,
# )
# model.fit(images, labels)
