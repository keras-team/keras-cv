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
import numpy as np
import tensorflow as tf

from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)


class RandomAddLayer(BaseImageAugmentationLayer):
    def __init__(self, value_range=(0.0, 1.0), fixed_value=None, **kwargs):
        super().__init__(**kwargs)
        self.value_range = value_range
        self.fixed_value = fixed_value

    def get_random_transformation(self, **kwargs):
        if self.fixed_value:
            return self.fixed_value
        return self._random_generator.random_uniform(
            [], minval=self.value_range[0], maxval=self.value_range[1]
        )

    def augment_image(self, image, transformation, **kwargs):
        return image + transformation

    def augment_label(self, label, transformation, **kwargs):
        return label + transformation

    def augment_bounding_boxes(self, bounding_boxes, transformation, **kwargs):
        return bounding_boxes + transformation

    def augment_keypoints(self, keypoints, transformation, **kwargs):
        return keypoints + transformation


class VectorizeDisabledLayer(BaseImageAugmentationLayer):
    def __init__(self, **kwargs):
        self.auto_vectorize = False
        super().__init__(**kwargs)


class BaseImageAugmentationLayerTest(tf.test.TestCase):
    def test_augment_single_image(self):
        add_layer = RandomAddLayer(fixed_value=2.0)
        image = np.random.random(size=(8, 8, 3)).astype("float32")
        output = add_layer(image)

        self.assertAllClose(image + 2.0, output)

    def test_augment_dict_return_type(self):
        add_layer = RandomAddLayer(fixed_value=2.0)
        image = np.random.random(size=(8, 8, 3)).astype("float32")
        output = add_layer({"images": image})

        self.assertIsInstance(output, dict)

    def test_auto_vectorize_disabled(self):
        vectorize_disabled_layer = VectorizeDisabledLayer()
        self.assertFalse(vectorize_disabled_layer.auto_vectorize)
        self.assertEqual(vectorize_disabled_layer._map_fn, tf.map_fn)

    def test_augment_casts_dtypes(self):
        add_layer = RandomAddLayer(fixed_value=2.0)
        images = tf.ones((2, 8, 8, 3), dtype="uint8")
        output = add_layer(images)

        self.assertAllClose(tf.ones((2, 8, 8, 3), dtype="float32") * 3.0, output)

    def test_augment_batch_images(self):
        add_layer = RandomAddLayer()
        images = np.random.random(size=(2, 8, 8, 3)).astype("float32")
        output = add_layer(images)

        diff = output - images
        # Make sure the first image and second image get different augmentation
        self.assertNotAllClose(diff[0], diff[1])

    def test_augment_image_and_label(self):
        add_layer = RandomAddLayer(fixed_value=2.0)
        image = np.random.random(size=(8, 8, 3)).astype("float32")
        label = np.random.random(size=(1,)).astype("float32")

        output = add_layer({"images": image, "labels": label})
        expected_output = {"images": image + 2.0, "labels": label + 2.0}
        self.assertAllClose(output, expected_output)

    def test_augment_image_and_target(self):
        add_layer = RandomAddLayer(fixed_value=2.0)
        image = np.random.random(size=(8, 8, 3)).astype("float32")
        label = np.random.random(size=(1,)).astype("float32")

        output = add_layer({"images": image, "targets": label})
        expected_output = {"images": image + 2.0, "targets": label + 2.0}
        self.assertAllClose(output, expected_output)

    def test_augment_batch_images_and_labels(self):
        add_layer = RandomAddLayer()
        images = np.random.random(size=(2, 8, 8, 3)).astype("float32")
        labels = np.random.random(size=(2, 1)).astype("float32")
        output = add_layer({"images": images, "labels": labels})

        image_diff = output["images"] - images
        label_diff = output["labels"] - labels
        # Make sure the first image and second image get different augmentation
        self.assertNotAllClose(image_diff[0], image_diff[1])
        self.assertNotAllClose(label_diff[0], label_diff[1])

    def test_augment_leaves_extra_dict_entries_unmodified(self):
        add_layer = RandomAddLayer(fixed_value=0.5)
        images = np.random.random(size=(8, 8, 3)).astype("float32")
        filenames = tf.constant("/path/to/first.jpg")
        inputs = {"images": images, "filenames": filenames}

        outputs = add_layer(inputs)

        self.assertListEqual(list(inputs.keys()), list(outputs.keys()))
        self.assertAllEqual(inputs["filenames"], outputs["filenames"])
        self.assertNotAllClose(inputs["images"], outputs["images"])
        self.assertAllEqual(inputs["images"], images)  # Assert original unchanged

    def test_augment_leaves_batched_extra_dict_entries_unmodified(self):
        add_layer = RandomAddLayer(fixed_value=0.5)
        images = np.random.random(size=(2, 8, 8, 3)).astype("float32")
        filenames = tf.constant(["/path/to/first.jpg", "/path/to/second.jpg"])
        inputs = {"images": images, "filenames": filenames}

        outputs = add_layer(inputs)

        self.assertListEqual(list(inputs.keys()), list(outputs.keys()))
        self.assertAllEqual(inputs["filenames"], outputs["filenames"])
        self.assertNotAllClose(inputs["images"], outputs["images"])
        self.assertAllEqual(inputs["images"], images)  # Assert original unchanged

    def test_augment_image_and_localization_data(self):
        add_layer = RandomAddLayer(fixed_value=2.0)
        images = np.random.random(size=(8, 8, 3)).astype("float32")
        bounding_boxes = np.random.random(size=(3, 4)).astype("float32")
        keypoints = np.random.random(size=(3, 5, 2)).astype("float32")

        output = add_layer(
            {"images": images, "bounding_boxes": bounding_boxes, "keypoints": keypoints}
        )

        expected_output = {
            "images": images + 2.0,
            "bounding_boxes": bounding_boxes + 2.0,
            "keypoints": keypoints + 2.0,
        }
        self.assertAllClose(output, expected_output)

    def test_augment_batch_image_and_localization_data(self):
        add_layer = RandomAddLayer()
        images = np.random.random(size=(2, 8, 8, 3)).astype("float32")
        bounding_boxes = np.random.random(size=(2, 3, 4)).astype("float32")
        keypoints = np.random.random(size=(2, 3, 5, 2)).astype("float32")

        output = add_layer(
            {"images": images, "bounding_boxes": bounding_boxes, "keypoints": keypoints}
        )

        bounding_boxes_diff = output["bounding_boxes"] - bounding_boxes
        keypoints_diff = output["keypoints"] - keypoints
        self.assertNotAllClose(bounding_boxes_diff[0], bounding_boxes_diff[1])
        self.assertNotAllClose(keypoints_diff[0], keypoints_diff[1])
