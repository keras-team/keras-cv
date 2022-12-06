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
import tensorflow_datasets as tfds
from waymo_open_dataset import dataset_pb2

from keras_cv.datasets.waymo import build_tensors_for_augmentation
from keras_cv.datasets.waymo import load
from keras_cv.datasets.waymo import transformer

TRAINING_RECORD_PATH = "/home/overflow/code/wod_records"  # "gs://waymo_open_dataset_v_1_0_0_individual_files/training"
TRANSFORMED_RECORD_PATH = "/home/overflow/code/wod_transformed"  # "gs://waymo_open_dataset_v_1_0_0_individual_files/training"


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

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


# Load the training dataset
filenames = tf.data.TFRecordDataset.list_files(
    os.path.join(TRAINING_RECORD_PATH, "*.tfrecord")
)


def convert_frame_generator(inputs):
    frame = dataset_pb2.Frame()
    frame.ParseFromString(inputs)
    return transformer.build_tensors_from_wod_frame(frame)


def _generate_frames(segments, transformer):
    def _generator():
        for record in tfds.as_numpy(segments):
            frame = dataset_pb2.Frame()
            frame.ParseFromString(record)
            yield transformer(frame)

    return _generator


for filename in filenames:
    train_ds = tf.data.TFRecordDataset(filename)
    train_ds = tf.data.Dataset.from_generator(
        _generate_frames(train_ds, transformer.build_tensors_from_wod_frame),
        output_signature=transformer.WOD_FRAME_OUTPUT_SIGNATURE,
    )
    train_ds = train_ds.map(
        build_tensors_for_augmentation, num_parallel_calls=tf.data.AUTOTUNE
    )
    start = time.time()
    step = 0
    transformed_filename = os.path.join(
        TRANSFORMED_RECORD_PATH, os.path.basename(filename.numpy().decode("utf-8"))
    )
    with tf.io.TFRecordWriter(transformed_filename) as writer:
        for examples in train_ds:
            serialized_example = serialize_example(
                examples["point_clouds"], examples["bounding_boxes"]
            )
            writer.write(serialized_example)
            step += 1
        print(f"Number of samples {step}")
    print(f"Time elapsed: {time.time()-start} seconds")
