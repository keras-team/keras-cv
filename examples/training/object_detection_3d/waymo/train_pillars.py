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
import time

import tensorflow as tf

from keras_cv.datasets.waymo import build_tensors_for_augmentation
from keras_cv.datasets.waymo import load
from keras_cv.layers import preprocessing3d

TRAINING_RECORD_PATH = (
    "./wod-records"  # "gs://waymo_open_dataset_v_1_0_0_individual_files/training"
)

# Load the training dataset
train_ds = load(TRAINING_RECORD_PATH)
# Batch by 1 to add a dimension for `num_frames`
train_ds = train_ds.map(build_tensors_for_augmentation).batch(1)

# Augment the training data
AUGMENTATION_LAYERS = [
    preprocessing3d.GlobalRandomFlipY(),
    preprocessing3d.GlobalRandomDroppingPoints(drop_rate=0.02),
    preprocessing3d.GlobalRandomRotation(max_rotation_angle_x=3.14),
    preprocessing3d.GlobalRandomScaling(scaling_factor_z=(0.5, 1.5)),
]


@tf.function
def augment(inputs):
    for layer in AUGMENTATION_LAYERS:
        inputs = layer(inputs)
    return inputs


train_ds = train_ds.map(augment)

# Very basic benchmarking
start = time.time()
_ = [None for x in train_ds.take(100)]
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
