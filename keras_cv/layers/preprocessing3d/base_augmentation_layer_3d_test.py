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

from keras_cv.layers.preprocessing3d import base_augmentation_layer_3d

POINT_CLOUDS = base_augmentation_layer_3d.POINT_CLOUDS
BOUNDING_BOXES = base_augmentation_layer_3d.BOUNDING_BOXES


class RandomAddLayer(base_augmentation_layer_3d.BaseAugmentationLayer3D):
    def __init__(self, translate_noise=(0.0, 0.0, 0.0), **kwargs):
        super().__init__(**kwargs)
        self._translate_noise = translate_noise

    def get_random_transformation(self, **kwargs):
        random_x = self._random_generator.random_normal(
            (), mean=0.0, stddev=self._translate_noise[0]
        )
        random_y = self._random_generator.random_normal(
            (), mean=0.0, stddev=self._translate_noise[1]
        )
        random_z = self._random_generator.random_normal(
            (), mean=0.0, stddev=self._translate_noise[2]
        )

        return {"pose": tf.stack([random_x, random_y, random_z, 0, 0, 0], axis=0)}

    def augment_point_clouds_bounding_boxes(
        self, point_clouds, bounding_boxes, transformation, **kwargs
    ):
        point_clouds_xyz = point_clouds[..., :3]
        point_clouds_xyz += transformation["pose"][:3]
        bounding_boxes_xyz = bounding_boxes[..., :3]
        bounding_boxes_xyz += transformation["pose"][:3]
        return (
            tf.concat([point_clouds_xyz, point_clouds[..., 3:]], axis=-1),
            tf.concat([bounding_boxes_xyz, bounding_boxes[..., 3:]], axis=-1),
        )


class VectorizeDisabledLayer(base_augmentation_layer_3d.BaseAugmentationLayer3D):
    def __init__(self, **kwargs):
        self.auto_vectorize = False
        super().__init__(**kwargs)


class BaseImageAugmentationLayerTest(tf.test.TestCase):
    def test_auto_vectorize_disabled(self):
        vectorize_disabled_layer = VectorizeDisabledLayer()
        self.assertFalse(vectorize_disabled_layer.auto_vectorize)
        self.assertEqual(vectorize_disabled_layer._map_fn, tf.map_fn)

    def test_augment_point_clouds_and_bounding_boxes(self):
        add_layer = RandomAddLayer(translate_noise=(1.0, 1.0, 1.0))
        point_clouds = np.random.random(size=(2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertNotAllClose(inputs, outputs)

    def test_augment_batch_point_clouds_and_bounding_boxes(self):
        add_layer = RandomAddLayer(translate_noise=(1.0, 1.0, 1.0))
        point_clouds = np.random.random(size=(3, 2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(3, 2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertNotAllClose(inputs, outputs)

    def test_augment_leaves_extra_dict_entries_unmodified(self):
        add_layer = RandomAddLayer(translate_noise=(1.0, 1.0, 1.0))
        point_clouds = np.random.random(size=(2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(2, 10, 7)).astype("float32")
        dummy = np.random.random(size=(2, 10, 7)).astype("float32")
        inputs = {
            POINT_CLOUDS: point_clouds,
            BOUNDING_BOXES: bounding_boxes,
            "dummy": dummy,
        }
        outputs = add_layer(inputs)

        self.assertAllEqual(inputs["dummy"], outputs["dummy"])
        self.assertNotAllClose(inputs, outputs)

    def test_augment_leaves_batched_extra_dict_entries_unmodified(self):
        add_layer = RandomAddLayer(translate_noise=(1.0, 1.0, 1.0))
        point_clouds = np.random.random(size=(3, 2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(3, 2, 10, 7)).astype("float32")
        dummy = np.random.random(size=(3, 2, 10, 7)).astype("float32")
        inputs = {
            POINT_CLOUDS: point_clouds,
            BOUNDING_BOXES: bounding_boxes,
            "dummy": dummy,
        }
        outputs = add_layer(inputs)

        self.assertAllEqual(inputs["dummy"], outputs["dummy"])
        self.assertNotAllClose(inputs, outputs)
