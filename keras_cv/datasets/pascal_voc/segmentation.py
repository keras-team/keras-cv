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

"""Data loader for Pascal VOC 2012 segmentation dataset.

The image classification and object detection (bounding box) data is covered by existing
TF datasets in https://www.tensorflow.org/datasets/catalog/voc. The segmentation data (
both class segmentation and instance segmentation) are included in the VOC 2012, but not
offered by TF-DS yet. This module is trying to fill this gap while TFDS team can
address this feature (b/252870855, https://github.com/tensorflow/datasets/issues/27 and
https://github.com/tensorflow/datasets/pull/1198).

The schema design is similar to the existing design of TFDS, but trimmed to fit the need
of Keras CV models.

This module contains following functionalities:

1. Download and unpack original data from Pascal VOC.
2. Reprocess and build up dataset that include image, class label, object bounding boxes,
   class and instance segmentation masks.
3. Produce tfrecords from the dataset.
4. Load existing tfrecords from result in 3.
"""
import logging
import multiprocessing
import os.path
import tarfile
import xml

import numpy as np
import tensorflow as tf
from PIL import Image

DATA_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"

# Note that this list doesn't contain the background class. In the classification use
# case, the label is 0 based (aeroplane -> 0), whereas in segmentation use case, the 0 is
# reserved for background, so aeroplane maps to 1.
CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
# This is used to map between string class to index.
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}


def _download_pascal_voc_2012(data_url, local_dir_path=None, override_extract=False):
    """Fetch the original Pascal VOC 2012 from remote URL.

    Args:
        data_url: string, the URL for the Pascal VOC data, should be in a tar package.
        local_dir_path: string, the local directory path to save the data.
    Returns:
        the path to the folder of extracted Pascal VOC data.
    """
    if not local_dir_path:
        fname = "pascal_voc_2012/data.tar"
    else:
        # Make sure the directory exists
        if not os.path.exists(local_dir_path):
            os.makedirs(local_dir_path, exist_ok=True)
        fname = os.path.join(local_dir_path, "data.tar")
    data_file_path = tf.keras.utils.get_file(fname=fname, origin=data_url)
    logging.info("Received data file from %s", data_file_path)
    # Extra the data into the same directory as the tar file.
    data_directory = os.path.dirname(data_file_path)
    # Note that the extracted data will be located in a folder `VOCdevkit` (from tar).
    # If the folder is already there and `override_extract` is False, then we will skip
    # extracting the folder again.
    if override_extract or not os.path.exists(
        os.path.join(data_directory, "VOCdevkit")
    ):
        logging.info("Extract data into %s", data_directory)
        with tarfile.open(data_file_path) as f:
            f.extractall(data_directory)
    return os.path.join(data_directory, "VOCdevkit", "VOC2012")


def _parse_annotation_data(annotation_file_path):
    """Parse the annotation XML file for the image.

    The annotation contains the metadata, as well as the object bounding box information.

    """
    with tf.io.gfile.GFile(annotation_file_path, "r") as f:
        root = xml.etree.ElementTree.parse(f).getroot()

        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        objects = []
        for obj in root.findall("object"):
            # Get object's label name.
            label = CLASS_TO_INDEX[obj.find("name").text.lower()]
            # Get objects' pose name.
            pose = obj.find("pose").text.lower()
            is_truncated = obj.find("truncated").text == "1"
            is_difficult = obj.find("difficult").text == "1"
            bndbox = obj.find("bndbox")
            xmax = int(bndbox.find("xmax").text)
            xmin = int(bndbox.find("xmin").text)
            ymax = int(bndbox.find("ymax").text)
            ymin = int(bndbox.find("ymin").text)
            objects.append(
                {
                    "label": label,
                    "pose": pose,
                    "bbox": [ymin, xmin, ymax, xmax],
                    "is_truncated": is_truncated,
                    "is_difficult": is_difficult,
                }
            )

        return {"width": width, "height": height, "objects": objects}


def _get_image_ids(data_dir, split):
    data_file_mapping = {"train": "train.txt", "eval": "val.txt", None: "trainval.txt"}
    with tf.io.gfile.GFile(
        os.path.join(data_dir, "ImageSets", "Segmentation", data_file_mapping[split]),
        "r",
    ) as f:
        image_ids = f.read().splitlines()
        logging.info(f"Received {len(image_ids)} images for {split} dataset.")
        return image_ids


def _convert_pascal_voc_segmentation_mask_encoding(
    segmentation_file_path, use_cache=True
):
    """Convert the original segmentation PNG file to proper encoding for tf.io API.

    The original PNG was in a 2D (without channel dimension), which will get a wrong value
    when directly read by tf.io API. In this function, the original image will be first
    read with PIL.image API, expand the last dimention and save back by TF API. This will
    ensure the converted image can be properly consumed by TF API.

    If a file with `xxx_converted.png` already exists, and `use_cache` is True, the
    conversion will be skipped.

    Args:
        segmentation_file_path: string, the original PNG file path from the dataset.
        override: boolean, whether to use the existing cached file result.
    Returns:
        the file path for converted PNG file, it will be suffixed with '_converted'.
    """
    dir_and_file_name, _ = os.path.splitext(segmentation_file_path)
    updated_file_path = dir_and_file_name + "_updated.png"
    if os.path.exists(updated_file_path) and use_cache:
        return updated_file_path

    with tf.io.gfile.GFile(segmentation_file_path, "rb") as f:
        mask = Image.open(f)

    original_mask = np.expand_dims(np.asarray(mask), axis=-1)
    # # Write the mask to PNG via the TF io API, so that it can be read properly.
    encoded_png = tf.io.encode_png(original_mask)
    tf.io.write_file(updated_file_path, encoded_png)
    return updated_file_path


def _parse_single_image(image_file_path):
    data_dir, image_file_name = os.path.split(image_file_path)
    data_dir = os.path.normpath(os.path.join(data_dir, os.path.pardir))
    image_id, _ = os.path.splitext(image_file_name)
    class_segmentation_file_path = os.path.join(
        data_dir, "SegmentationClass", image_id + ".png"
    )
    class_segmentation_file_path = _convert_pascal_voc_segmentation_mask_encoding(
        class_segmentation_file_path
    )
    object_segmentation_file_path = os.path.join(
        data_dir, "SegmentationObject", image_id + ".png"
    )
    object_segmentation_file_path = _convert_pascal_voc_segmentation_mask_encoding(
        object_segmentation_file_path
    )
    annotation_file_path = os.path.join(data_dir, "Annotations", image_id + ".xml")
    image_annotations = _parse_annotation_data(annotation_file_path)

    result = {
        "image/filename": image_id + ".jpg",
        "image/file_path": image_file_path,
        "segmentation/class/file_path": class_segmentation_file_path,
        "segmentation/object/file_path": object_segmentation_file_path,
    }
    result.update(image_annotations)
    # Labels field should be same as the 'object.label'
    labels = list(set([o["label"] for o in result["objects"]]))
    result["labels"] = sorted(labels)
    return result


def _build_metadata(data_dir, image_ids):
    # Parallel process all the images.
    image_file_paths = [
        os.path.join(data_dir, "JPEGImages", i + ".jpg") for i in image_ids
    ]
    pool_size = 10 if len(image_ids) > 10 else len(image_ids)
    with multiprocessing.Pool(pool_size) as p:
        metadata = p.map(_parse_single_image, image_file_paths)

    # Transpose the metadata which convert from list of dict to dict of list.
    keys = [
        "image/filename",
        "image/file_path",
        "segmentation/class/file_path",
        "segmentation/object/file_path",
        "labels",
        "width",
        "height",
    ]
    result = {}
    for key in keys:
        values = [value[key] for value in metadata]
        result[key] = values

    # The ragged objects need some special handling
    for key in ["label", "pose", "bbox", "is_truncated", "is_difficult"]:
        values = []
        objects = [value["objects"] for value in metadata]
        for object in objects:
            values.append([o[key] for o in object])
        result["objects/" + key] = values
    return result


def _load_images(example):
    image_file_path = example.pop("image/file_path")
    segmentation_class_file_path = example.pop("segmentation/class/file_path")
    segmentation_object_file_path = example.pop("segmentation/object/file_path")
    image = tf.io.read_file(image_file_path)
    image = tf.image.decode_jpeg(image)

    segmentation_class_mask = tf.io.read_file(segmentation_class_file_path)
    segmentation_class_mask = tf.image.decode_png(segmentation_class_mask)

    segmentation_object_mask = tf.io.read_file(segmentation_object_file_path)
    segmentation_object_mask = tf.image.decode_png(segmentation_object_mask)

    example.update(
        {
            "image": image,
            "class_segmentation": segmentation_class_mask,
            "object_segmentation": segmentation_object_mask,
        }
    )
    return example


def _build_dataset_from_metadata(metadata):
    # The objects need some manual conversion to ragged tensor.
    metadata["labels"] = tf.ragged.constant(metadata["labels"])
    metadata["objects/label"] = tf.ragged.constant(metadata["objects/label"])
    metadata["objects/pose"] = tf.ragged.constant(metadata["objects/pose"])
    metadata["objects/is_truncated"] = tf.ragged.constant(
        metadata["objects/is_truncated"]
    )
    metadata["objects/is_difficult"] = tf.ragged.constant(
        metadata["objects/is_difficult"]
    )
    metadata["objects/bbox"] = tf.ragged.constant(
        metadata["objects/bbox"], ragged_rank=1
    )

    dataset = tf.data.Dataset.from_tensor_slices(metadata)
    dataset = dataset.map(_load_images, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


def load(
    split="train",
    data_dir=None,
):
    """Load the Pacal VOC 2012 dataset.

    This function will download the data tar file from remote if needed, and untar to
    the local `data_dir`, and build dataset from it.

    Args:
        split: string, can be 'train', 'eval', or None. When None, both train and eval data
            will be loaded. Default to `train`
        data_dir: string, local directory path for the loaded data. This will be used to
            download the data file, and unzip. It will be used as a cach directory.
            Default to None, and `~/.keras/pascal_voc_2012` will be used.
    """
    supported_split_value = ["train", "eval", None]
    if split not in supported_split_value:
        raise ValueError(
            f"The support value for `split` are {supported_split_value}. "
            f"Got: {split}"
        )

    if data_dir is not None:
        data_dir = os.path.expanduser(data_dir)

    data_dir = _download_pascal_voc_2012(DATA_URL, local_dir_path=data_dir)
    image_ids = _get_image_ids(data_dir, split)
    metadata = _build_metadata(data_dir, image_ids)
    dataset = _build_dataset_from_metadata(metadata)

    return dataset
