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

The image classification and object detection (bounding box) data is covered by
existing TF datasets in https://www.tensorflow.org/datasets/catalog/voc. The
segmentation data (both class segmentation and instance segmentation) are
included in the VOC 2012, but not offered by TF-DS yet. This module is trying to
fill this gap while TFDS team can address this feature (b/252870855,
https://github.com/tensorflow/datasets/issues/27 and
https://github.com/tensorflow/datasets/pull/1198).

The schema design is similar to the existing design of TFDS, but trimmed to fit
the need of Keras CV models.

This module contains following functionalities:

1. Download and unpack original data from Pascal VOC.
2. Reprocess and build up dataset that include image, class label, object
   bounding boxes,
   class and instance segmentation masks.
3. Produce tfrecords from the dataset.
4. Load existing tfrecords from result in 3.
"""

import logging
import multiprocessing
import os.path
import random
import tarfile
import xml

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

from keras_cv.src.api_export import keras_cv_export

VOC_URL = "https://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"  # noqa: E501

"""
@InProceedings{{BharathICCV2011,
    author = "Bharath Hariharan and Pablo Arbelaez and Lubomir Bourdev and Subhransu Maji and Jitendra Malik",
    title = "Semantic Contours from Inverse Detectors",
    booktitle = "International Conference on Computer Vision (ICCV)",
    year = "2011"}}
"""  # noqa: E501
SBD_URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz"  # noqa: E501


# Note that this list doesn't contain the background class. In the
# classification use case, the label is 0 based (aeroplane -> 0), whereas in
# segmentation use case, the 0 is reserved for background, so aeroplane maps to
# 1.
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

# For the mask data in the PNG file, the encoded raw pixel value need to be
# converted to the proper class index. In the following map, [0, 0, 0] will be
# convert to 0, and [128, 0, 0] will be converted to 1, so on so forth. Also
# note that the mask class is 1 base since class 0 is reserved for the
# background. The [128, 0, 0] (class 1) is mapped to `aeroplane`.
VOC_PNG_COLOR_VALUE = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]
# Will be populated by _maybe_populate_voc_color_mapping() below.
VOC_PNG_COLOR_MAPPING = None


def _maybe_populate_voc_color_mapping():
    # Lazy creation of VOC_PNG_COLOR_MAPPING, which could take 64M memory.
    global VOC_PNG_COLOR_MAPPING
    if VOC_PNG_COLOR_MAPPING is None:
        VOC_PNG_COLOR_MAPPING = [0] * (256**3)
        for i, colormap in enumerate(VOC_PNG_COLOR_VALUE):
            VOC_PNG_COLOR_MAPPING[
                (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]
            ] = i
        # There is a special mapping with [224, 224, 192] -> 255
        VOC_PNG_COLOR_MAPPING[224 * 256 * 256 + 224 * 256 + 192] = 255
        VOC_PNG_COLOR_MAPPING = tf.constant(VOC_PNG_COLOR_MAPPING)
    return VOC_PNG_COLOR_MAPPING


def _download_data_file(
    data_url, extracted_dir, local_dir_path=None, override_extract=False
):
    """Fetch the original VOC or Semantic Boundaries Dataset from remote URL.

    Args:
        data_url: string, the URL for the data to be downloaded, should be in a
            zipped tar package.
        local_dir_path: string, the local directory path to save the data.
    Returns:
        the path to the folder of extracted data.
    """
    if not local_dir_path:
        # download to ~/.keras/datasets/fname
        cache_dir = os.path.join(os.path.expanduser("~"), ".keras/datasets")
        fname = os.path.join(cache_dir, os.path.basename(data_url))
    else:
        # Make sure the directory exists
        if not os.path.exists(local_dir_path):
            os.makedirs(local_dir_path, exist_ok=True)
        # download to local_dir_path/fname
        fname = os.path.join(local_dir_path, os.path.basename(data_url))
    data_directory = os.path.join(os.path.dirname(fname), extracted_dir)
    if not override_extract and os.path.exists(data_directory):
        logging.info("data directory %s already exist", data_directory)
        return data_directory
    data_file_path = keras.utils.get_file(fname=fname, origin=data_url)
    # Extra the data into the same directory as the tar file.
    data_directory = os.path.dirname(data_file_path)
    logging.info("Extract data into %s", data_directory)
    with tarfile.open(data_file_path) as f:
        f.extractall(data_directory)
    return os.path.join(data_directory, extracted_dir)


def _parse_annotation_data(annotation_file_path):
    """Parse the annotation XML file for the image.

    The annotation contains the metadata, as well as the object bounding box
    information.

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
    data_file_mapping = {
        "train": "train.txt",
        "eval": "val.txt",
        "trainval": "trainval.txt",
        # TODO(tanzhenyu): add diff dataset
        # "diff": "diff.txt",
    }
    with tf.io.gfile.GFile(
        os.path.join(
            data_dir, "ImageSets", "Segmentation", data_file_mapping[split]
        ),
        "r",
    ) as f:
        image_ids = f.read().splitlines()
        logging.info(f"Received {len(image_ids)} images for {split} dataset.")
        return image_ids


def _get_sbd_image_ids(data_dir, split):
    data_file_mapping = {"sbd_train": "train.txt", "sbd_eval": "val.txt"}
    with tf.io.gfile.GFile(
        os.path.join(data_dir, data_file_mapping[split]),
        "r",
    ) as f:
        image_ids = f.read().splitlines()
        logging.info(f"Received {len(image_ids)} images for {split} dataset.")
        return image_ids


def _parse_single_image(image_file_path):
    data_dir, image_file_name = os.path.split(image_file_path)
    data_dir = os.path.normpath(os.path.join(data_dir, os.path.pardir))
    image_id, _ = os.path.splitext(image_file_name)
    class_segmentation_file_path = os.path.join(
        data_dir, "SegmentationClass", image_id + ".png"
    )
    object_segmentation_file_path = os.path.join(
        data_dir, "SegmentationObject", image_id + ".png"
    )
    annotation_file_path = os.path.join(
        data_dir, "Annotations", image_id + ".xml"
    )
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


def _parse_single_sbd_image(image_file_path):
    data_dir, image_file_name = os.path.split(image_file_path)
    data_dir = os.path.normpath(os.path.join(data_dir, os.path.pardir))
    image_id, _ = os.path.splitext(image_file_name)
    class_segmentation_file_path = os.path.join(
        data_dir, "cls", image_id + ".mat"
    )
    object_segmentation_file_path = os.path.join(
        data_dir, "inst", image_id + ".mat"
    )
    result = {
        "image/filename": image_id + ".jpg",
        "image/file_path": image_file_path,
        "segmentation/class/file_path": class_segmentation_file_path,
        "segmentation/object/file_path": object_segmentation_file_path,
    }
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


def _build_sbd_metadata(data_dir, image_ids):
    # Parallel process all the images.
    image_file_paths = [
        os.path.join(data_dir, "img", i + ".jpg") for i in image_ids
    ]
    pool_size = 10 if len(image_ids) > 10 else len(image_ids)
    with multiprocessing.Pool(pool_size) as p:
        metadata = p.map(_parse_single_sbd_image, image_file_paths)

    keys = [
        "image/filename",
        "image/file_path",
        "segmentation/class/file_path",
        "segmentation/object/file_path",
    ]
    result = {}
    for key in keys:
        values = [value[key] for value in metadata]
        result[key] = values
    return result


# With jit_compile=True, there will be 0.4 sec compilation overhead, but save
# about 0.2 sec per 1000 images. See
# https://github.com/keras-team/keras-cv/pull/943#discussion_r1001092882
# for more details.
@tf.function(jit_compile=True)
def _decode_png_mask(mask):
    """Decode the raw PNG image and convert it to 2D tensor with probably
    class."""
    # Cast the mask to int32 since the original uint8 will overflow when
    # multiplied with 256
    mask = tf.cast(mask, tf.int32)
    mask = mask[:, :, 0] * 256 * 256 + mask[:, :, 1] * 256 + mask[:, :, 2]
    mask = tf.expand_dims(tf.gather(VOC_PNG_COLOR_MAPPING, mask), -1)
    mask = tf.cast(mask, tf.uint8)
    return mask


def _load_images(example):
    image_file_path = example.pop("image/file_path")
    segmentation_class_file_path = example.pop("segmentation/class/file_path")
    segmentation_object_file_path = example.pop("segmentation/object/file_path")
    image = tf.io.read_file(image_file_path)
    image = tf.image.decode_jpeg(image)

    segmentation_class_mask = tf.io.read_file(segmentation_class_file_path)
    segmentation_class_mask = tf.image.decode_png(segmentation_class_mask)
    segmentation_class_mask = _decode_png_mask(segmentation_class_mask)

    segmentation_object_mask = tf.io.read_file(segmentation_object_file_path)
    segmentation_object_mask = tf.image.decode_png(segmentation_object_mask)
    segmentation_object_mask = _decode_png_mask(segmentation_object_mask)

    example.update(
        {
            "image": image,
            "class_segmentation": segmentation_class_mask,
            "object_segmentation": segmentation_object_mask,
        }
    )
    return example


def _load_sbd_images(image_file_path, seg_cls_file_path, seg_obj_file_path):
    image = tf.io.read_file(image_file_path)
    image = tf.image.decode_jpeg(image)

    segmentation_class_mask = tfds.core.lazy_imports.scipy.io.loadmat(
        seg_cls_file_path
    )
    segmentation_class_mask = segmentation_class_mask["GTcls"]["Segmentation"][
        0
    ][0]
    segmentation_class_mask = segmentation_class_mask[..., np.newaxis]

    segmentation_object_mask = tfds.core.lazy_imports.scipy.io.loadmat(
        seg_obj_file_path
    )
    segmentation_object_mask = segmentation_object_mask["GTinst"][
        "Segmentation"
    ][0][0]
    segmentation_object_mask = segmentation_object_mask[..., np.newaxis]

    return {
        "image": image,
        "class_segmentation": segmentation_class_mask,
        "object_segmentation": segmentation_object_mask,
    }


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


def _build_sbd_dataset_from_metadata(metadata):
    img_filepath = metadata["image/file_path"]
    cls_filepath = metadata["segmentation/class/file_path"]
    obj_filepath = metadata["segmentation/object/file_path"]

    def md_gen():
        c = list(zip(img_filepath, cls_filepath, obj_filepath))
        # random shuffling for each generator boosts up the quality.
        random.shuffle(c)
        for fp in c:
            img_fp, cls_fp, obj_fp = fp
            yield _load_sbd_images(img_fp, cls_fp, obj_fp)

    dataset = tf.data.Dataset.from_generator(
        md_gen,
        output_signature=(
            {
                "image": tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
                "class_segmentation": tf.TensorSpec(
                    shape=(None, None, 1), dtype=tf.uint8
                ),
                "object_segmentation": tf.TensorSpec(
                    shape=(None, None, 1), dtype=tf.uint8
                ),
            }
        ),
    )

    return dataset


@keras_cv_export(
    "keras_cv.datasets.pascal_voc.segmentation.load",
    package="keras_cv.datasets.pascal_voc_segmentation",
)
def load(
    split="sbd_train",
    data_dir=None,
):
    """Load the Pacal VOC 2012 dataset.

    This function will download the data tar file from remote if needed, and
    untar to the local `data_dir`, and build dataset from it.

    It supports both VOC2012 and Semantic Boundaries Dataset (SBD).

    The returned segmentation masks will be int ranging from [0, num_classes),
    as well as 255 which is the boundary mask.

    Args:
        split: string, can be 'train', 'eval', 'trainval", 'sbd_train', or
            'sbd_eval'. 'sbd_train' represents the training dataset for SBD
            dataset, while 'train' represents the training dataset for VOC2012
            dataset. Defaults to `sbd_train`.
        data_dir: string, local directory path for the loaded data. This will be
            used to download the data file, and unzip. It will be used as a
            cache directory. Defaults to None, and `~/.keras/pascal_voc_2012`
            will be used.
    """
    supported_split_value = [
        "train",
        "eval",
        "trainval",
        "sbd_train",
        "sbd_eval",
    ]
    if split not in supported_split_value:
        raise ValueError(
            f"The support value for `split` are {supported_split_value}. "
            f"Got: {split}"
        )

    if data_dir is not None:
        data_dir = os.path.expanduser(data_dir)

    if "sbd" in split:
        return _load_sbd(split, data_dir)
    else:
        return _load_voc(split, data_dir)


def _load_voc(
    split="train",
    data_dir=None,
):
    extracted_dir = os.path.join("VOCdevkit", "VOC2012")
    data_dir = _download_data_file(
        VOC_URL, extracted_dir=extracted_dir, local_dir_path=data_dir
    )
    image_ids = _get_image_ids(data_dir, split)
    # len(metadata) = #samples, metadata[i] is a dict.
    metadata = _build_metadata(data_dir, image_ids)
    _maybe_populate_voc_color_mapping()
    dataset = _build_dataset_from_metadata(metadata)

    return dataset


def _load_sbd(
    split="sbd_train",
    data_dir=None,
):
    extracted_dir = os.path.join("benchmark_RELEASE", "dataset")
    data_dir = _download_data_file(
        SBD_URL, extracted_dir=extracted_dir, local_dir_path=data_dir
    )
    image_ids = _get_sbd_image_ids(data_dir, split)
    # len(metadata) = #samples, metadata[i] is a dict.
    metadata = _build_sbd_metadata(data_dir, image_ids)
    dataset = _build_sbd_dataset_from_metadata(metadata)
    return dataset
