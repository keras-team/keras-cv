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
import pathlib
import sys

import tensorflow as tf
from absl import flags

from keras_cv.datasets.pascal_voc import segmentation

extracted_dir = os.path.join("VOCdevkit", "VOC2012")


class PascalVocSegmentationDataTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.tempdir = self.get_tempdir()
        # Note that this will not work with bazel, need to be rewrite into relying on
        # FLAGS.test_srcdir
        self.test_data_tar_path = os.path.abspath(
            os.path.join(
                os.path.abspath(__file__), os.path.pardir, "test_data", "VOC_mini.tar"
            )
        )

    def get_tempdir(self):
        try:
            flags.FLAGS.test_tmpdir
        except flags.UnparsedFlagAccessError:
            # Need to initialize flags when running `pytest`.
            flags.FLAGS(sys.argv, known_only=True)
        return self.create_tempdir().full_path

    def test_download_data(self):
        # Since the original data package is too large, we use a small package as a
        # replacement.
        local_data_dir = os.path.join(self.tempdir, "pascal_voc_2012/")
        test_data_dir = segmentation._download_data_file(
            data_url=pathlib.Path(self.test_data_tar_path).as_uri(),
            extracted_dir=extracted_dir,
            local_dir_path=local_data_dir,
        )

        self.assertTrue(os.path.exists(test_data_dir))
        # Make sure the data is unzipped correctly and populated with correct content
        expected_subdirs = [
            "Annotations",
            "ImageSets",
            "JPEGImages",
            "SegmentationClass",
            "SegmentationObject",
        ]
        for sub_dir in expected_subdirs:
            self.assertTrue(os.path.exists(os.path.join(test_data_dir, sub_dir)))

    def test_skip_download_and_override(self):
        local_data_dir = os.path.join(self.tempdir, "pascal_voc_2012/")
        test_data_dir = segmentation._download_data_file(
            data_url=pathlib.Path(self.test_data_tar_path).as_uri(),
            extracted_dir=extracted_dir,
            local_dir_path=local_data_dir,
        )

        # Touch a file in the test_data_dir and make sure it exists (not being override)
        # when invoke the _download_data_file again
        os.makedirs(os.path.join(test_data_dir, "Annotations", "dummy_dir"))
        segmentation._download_data_file(
            data_url=pathlib.Path(self.test_data_tar_path).as_uri(),
            extracted_dir=extracted_dir,
            local_dir_path=local_data_dir,
            override_extract=False,
        )
        self.assertTrue(
            os.path.exists(os.path.join(test_data_dir, "Annotations", "dummy_dir"))
        )

    def test_get_image_ids(self):
        local_data_dir = os.path.join(self.tempdir, "pascal_voc_2012/")
        data_dir = segmentation._download_data_file(
            data_url=pathlib.Path(self.test_data_tar_path).as_uri(),
            extracted_dir=extracted_dir,
            local_dir_path=local_data_dir,
        )
        train_ids = ["2007_000032", "2007_000039", "2007_000063"]
        eval_ids = ["2007_000033"]
        train_eval_ids = train_ids + eval_ids
        self.assertEquals(segmentation._get_image_ids(data_dir, "train"), train_ids)
        self.assertEquals(segmentation._get_image_ids(data_dir, "eval"), eval_ids)
        self.assertEquals(
            segmentation._get_image_ids(data_dir, "trainval"), train_eval_ids
        )

    def test_parse_annotation_file(self):
        local_data_dir = os.path.join(self.tempdir, "pascal_voc_2012/")
        data_dir = segmentation._download_data_file(
            data_url=pathlib.Path(self.test_data_tar_path).as_uri(),
            extracted_dir=extracted_dir,
            local_dir_path=local_data_dir,
        )
        # One of the train file.
        annotation_file = os.path.join(data_dir, "Annotations", "2007_000032.xml")
        metadata = segmentation._parse_annotation_data(annotation_file)
        expected_result = {
            "height": 281,
            "width": 500,
            "objects": [
                {
                    "label": 0,
                    "pose": "frontal",
                    "bbox": [78, 104, 183, 375],
                    "is_truncated": False,
                    "is_difficult": False,
                },
                {
                    "label": 0,
                    "pose": "left",
                    "bbox": [88, 133, 123, 197],
                    "is_truncated": False,
                    "is_difficult": False,
                },
                {
                    "label": 14,
                    "pose": "rear",
                    "bbox": [180, 195, 229, 213],
                    "is_truncated": False,
                    "is_difficult": False,
                },
                {
                    "label": 14,
                    "pose": "rear",
                    "bbox": [189, 26, 238, 44],
                    "is_truncated": False,
                    "is_difficult": False,
                },
            ],
        }
        self.assertEquals(metadata, expected_result)

    def test_decode_png_mask(self):
        local_data_dir = os.path.join(self.tempdir, "pascal_voc_2012/")
        data_dir = segmentation._download_data_file(
            data_url=pathlib.Path(self.test_data_tar_path).as_uri(),
            extracted_dir=extracted_dir,
            local_dir_path=local_data_dir,
        )
        mask_file = os.path.join(data_dir, "SegmentationClass", "2007_000032.png")
        mask = tf.io.decode_png(tf.io.read_file(mask_file))
        segmentation._maybe_populate_voc_color_mapping()
        mask = segmentation._decode_png_mask(mask)

        self.assertEquals(mask.shape, (281, 500, 1))
        self.assertEquals(tf.reduce_max(mask), 255)  # The 255 value is for the boundary
        self.assertEquals(tf.reduce_min(mask), 0)  # The 0 value is for the background
        # The mask contains two classes, 1 and 15, see the label section in the previous
        # test case.
        self.assertEquals(tf.reduce_sum(tf.cast(tf.equal(mask, 1), tf.int32)), 4734)
        self.assertEquals(tf.reduce_sum(tf.cast(tf.equal(mask, 15), tf.int32)), 866)

    def test_parse_single_image(self):
        local_data_dir = os.path.join(self.tempdir, "pascal_voc_2012/")
        data_dir = segmentation._download_data_file(
            data_url=pathlib.Path(self.test_data_tar_path).as_uri(),
            extracted_dir=extracted_dir,
            local_dir_path=local_data_dir,
        )
        image_file = os.path.join(data_dir, "JPEGImages", "2007_000032.jpg")
        result_dict = segmentation._parse_single_image(image_file)
        expected_result = {
            "image/filename": "2007_000032.jpg",
            "image/file_path": image_file,
            "height": 281,
            "width": 500,
            "objects": [
                {
                    "label": 0,
                    "pose": "frontal",
                    "bbox": [78, 104, 183, 375],
                    "is_truncated": False,
                    "is_difficult": False,
                },
                {
                    "label": 0,
                    "pose": "left",
                    "bbox": [88, 133, 123, 197],
                    "is_truncated": False,
                    "is_difficult": False,
                },
                {
                    "label": 14,
                    "pose": "rear",
                    "bbox": [180, 195, 229, 213],
                    "is_truncated": False,
                    "is_difficult": False,
                },
                {
                    "label": 14,
                    "pose": "rear",
                    "bbox": [189, 26, 238, 44],
                    "is_truncated": False,
                    "is_difficult": False,
                },
            ],
            "labels": [0, 14],
            "segmentation/class/file_path": os.path.join(
                data_dir, "SegmentationClass", "2007_000032.png"
            ),
            "segmentation/object/file_path": os.path.join(
                data_dir, "SegmentationObject", "2007_000032.png"
            ),
        }
        self.assertEquals(result_dict, expected_result)

    def test_build_metadata(self):
        local_data_dir = os.path.join(self.tempdir, "pascal_voc_2012/")
        data_dir = segmentation._download_data_file(
            data_url=pathlib.Path(self.test_data_tar_path).as_uri(),
            extracted_dir=extracted_dir,
            local_dir_path=local_data_dir,
        )
        image_ids = segmentation._get_image_ids(data_dir, "trainval")
        metadata = segmentation._build_metadata(data_dir, image_ids)

        self.assertEquals(
            metadata["image/filename"],
            [
                "2007_000032.jpg",
                "2007_000039.jpg",
                "2007_000063.jpg",
                "2007_000033.jpg",
            ],
        )
        expected_keys = [
            "image/filename",
            "image/file_path",
            "segmentation/class/file_path",
            "segmentation/object/file_path",
            "labels",
            "width",
            "height",
            "objects/label",
            "objects/pose",
            "objects/bbox",
            "objects/is_truncated",
            "objects/is_difficult",
        ]
        for key in expected_keys:
            self.assertLen(metadata[key], 4)

    def test_build_dataset(self):
        local_data_dir = os.path.join(self.tempdir, "pascal_voc_2012/")
        data_dir = segmentation._download_data_file(
            data_url=pathlib.Path(self.test_data_tar_path).as_uri(),
            extracted_dir=extracted_dir,
            local_dir_path=local_data_dir,
        )
        image_ids = segmentation._get_image_ids(data_dir, "train")
        metadata = segmentation._build_metadata(data_dir, image_ids)
        segmentation._maybe_populate_voc_color_mapping()
        dataset = segmentation._build_dataset_from_metadata(metadata)

        entry = next(dataset.take(1).as_numpy_iterator())
        self.assertEquals(entry["image/filename"], b"2007_000032.jpg")
        expected_keys = [
            "image",
            "image/filename",
            "labels",
            "width",
            "height",
            "objects/label",
            "objects/pose",
            "objects/bbox",
            "objects/is_truncated",
            "objects/is_difficult",
            "class_segmentation",
            "object_segmentation",
        ]
        for key in expected_keys:
            self.assertIn(key, entry)

        # Check the mask png content
        png = entry["class_segmentation"]
        self.assertEquals(png.shape, (281, 500, 1))
        self.assertEquals(tf.reduce_max(png), 255)  # The 255 value is for the boundary
        self.assertEquals(tf.reduce_min(png), 0)  # The 0 value is for the background
        # The mask contains two classes, 1 and 15, see the label section in the previous
        # test case.
        self.assertEquals(tf.reduce_sum(tf.cast(tf.equal(png, 1), tf.int32)), 4734)
        self.assertEquals(tf.reduce_sum(tf.cast(tf.equal(png, 15), tf.int32)), 866)
