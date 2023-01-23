# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time

import tensorflow as tf

from keras_cv.layers import preprocessing3d

# use serialize_records to convert WOD frame to Tensors
TRAINING_RECORD_PATH = (
    "./wod_transformed"  # "gs://waymo_open_dataset_v_1_0_0_individual_files/training"
)

global_batch = 1

features_dict = {
    "point_clouds": tf.io.RaggedFeature(dtype=tf.float32),
    "bounding_boxes": tf.io.RaggedFeature(dtype=tf.float32),
}


def build_tensors(x):
    res = {}
    x = tf.io.parse_example(x, features_dict)
    point_clouds = x["point_clouds"]
    boxes = x["bounding_boxes"]
    print("point cloud shape ", point_clouds.shape)
    print("box shape ", boxes.shape)
    point_clouds = tf.reshape(point_clouds, [1, -1, 8])
    boxes = tf.reshape(boxes, [1, -1, 11])
    res["point_clouds"] = point_clouds
    res["bounding_boxes"] = boxes
    return res


def pad_tensors(x):
    res = {}
    point_clouds = x["point_clouds"]
    boxes = x["bounding_boxes"]
    point_clouds = point_clouds.to_tensor(
        default_value=-1.0, shape=[global_batch, 1, 200000, 8]
    )
    boxes = boxes.to_tensor(default_value=-1.0, shape=[global_batch, 1, 1000, 11])
    res["point_clouds"] = point_clouds
    res["bounding_boxes"] = boxes
    return res


# Load the training dataset
filenames = tf.data.Dataset.list_files(os.path.join(TRAINING_RECORD_PATH, "*.tfrecord"))
train_ds = tf.data.TFRecordDataset(filenames)
train_ds = train_ds.map(build_tensors, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.apply(
    tf.data.experimental.dense_to_ragged_batch(global_batch, drop_remainder=True)
)
# Batch by 1 to add a dimension for `num_frames`
train_ds = train_ds.map(pad_tensors, num_parallel_calls=tf.data.AUTOTUNE)
print(f"train ds element spec {train_ds.element_spec}")

# Augment the training data
AUGMENTATION_LAYERS = [
    preprocessing3d.GlobalRandomFlip(),
    preprocessing3d.GlobalRandomDroppingPoints(drop_rate=0.02),
    preprocessing3d.GlobalRandomRotation(max_rotation_angle_x=3.14),
    preprocessing3d.GlobalRandomScaling(scaling_factor_z=(0.5, 1.5)),
    preprocessing3d.GroupPointsByBoundingBoxes(),
]


@tf.function
def augment(inputs):
    for layer in AUGMENTATION_LAYERS:
        inputs = layer(inputs)
    return inputs


train_ds = train_ds.map(augment)

# Very basic benchmarking
start = time.time()
step = 0
for examples in train_ds:
    step += 1
print(f"Number of batches {step}")
print(f"Time elapsed: {time.time()-start} seconds")

# Everything after this is not ready -- pending getting a model available
# in KerasCV

# ### Load the evaluation dataset
# EVALUATION_RECORD_PATH = "./wod-records"#"gs://waymo_open_dataset_v_1_0_0_individual_files/validation"
# eval_ds = load(EVALUATION_RECORD_PATH, simple_transformer, output_signature)
#
#
# ### Load and compile the model
# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():
#     model = None  # TODO Need to import model and instantiate it here
#
# model.compile(optimizer="adam", loss=None)  # TODO need to specify appropriate loss here
#
#
# ### Fit the model with a callback to log scores on our evaluation dataset
# model.fit(
#     train_ds,
#     callbacks=[
#         # TODO Uncomment when ready from keras_cv.callbacks import WaymoDetectionMetrics
#         WaymoDetectionMetrics(eval_ds),
#         keras.callbacks.TensorBoard(TENSORBOARD_LOGS_PATH),
#     ],
# )
