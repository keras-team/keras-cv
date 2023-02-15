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

from keras_cv.datasets.waymo import build_tensors_for_augmentation
from keras_cv.datasets.waymo import load

TRAINING_RECORD_PATH = "./wod_records"  # "gs://waymo_open_dataset_v_1_0_0_individual_files/training"
TRANSFORMED_RECORD_PATH = "./wod_transformed"  # "gs://waymo_open_dataset_v_1_0_0_individual_files/training"


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def serialize_example(feature0, feature1):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        "point_clouds": _float_feature(tf.reshape(feature0, [-1]).numpy()),
        "bounding_boxes": _float_feature(tf.reshape(feature1, [-1]).numpy()),
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )
    return example_proto.SerializeToString()


# Load the training dataset
filenames = os.listdir(TRAINING_RECORD_PATH)

for filename in filenames:
    train_ds = load([os.path.join(TRAINING_RECORD_PATH, filename)])
    train_ds = train_ds.map(
        build_tensors_for_augmentation, num_parallel_calls=tf.data.AUTOTUNE
    )
    start = time.time()
    step = 0
    transformed_filename = os.path.join(TRANSFORMED_RECORD_PATH, filename)
    with tf.io.TFRecordWriter(transformed_filename) as writer:
        for examples in train_ds:
            serialized_example = serialize_example(
                examples["point_clouds"], examples["bounding_boxes"]
            )
            writer.write(serialized_example)
            step += 1
        print(f"Number of samples {step}")
    print(f"Time elapsed: {time.time()-start} seconds")
