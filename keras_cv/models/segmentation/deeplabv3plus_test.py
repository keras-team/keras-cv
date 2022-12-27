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

import tensorflow as tf

from keras_cv import models
from keras_cv.models import segmentation


class DeeplabV3PlusTest(tf.test.TestCase):
    def test_deeplab_model_with_components(self):
        backbone = models.ResNet50V2(
            include_rescaling=True,
            stackwise_dilations=[1, 1, 1, 2],
            input_shape=(256, 256, 3),
            include_top=False,
            weights=None,
        )

        model = segmentation.DeepLabV3Plus(classes=11, backbone=backbone)

        input_image = tf.random.uniform(shape=[2, 256, 256, 3])
        output = model(input_image, training=True)

        self.assertEquals(output["output"].shape, [2, 256, 256, 11])

    def test_greyscale_input(self):
        backbone = models.ResNet50V2(
            include_rescaling=True,
            stackwise_dilations=[1, 1, 1, 2],
            input_shape=(64, 64, 1),
            include_top=False,
            weights=None,
        )
        model = segmentation.DeepLabV3Plus(classes=11, backbone=backbone)
        input_image = tf.random.uniform(shape=[1, 64, 64, 1])
        output = model(input_image, training=True)

        self.assertEquals(output["output"].shape, [1, 64, 64, 11])

    def test_missing_input_shape(self):
        with self.assertRaisesRegex(
            ValueError,
            "Input shapes for both the backbone and DeepLabV3Plus are `None`.",
        ):
            backbone = models.ResNet50V2(
                include_rescaling=True,
                include_top=False,
                stackwise_dilations=[1, 1, 1, 2],
            )
            segmentation.DeepLabV3Plus(classes=11, backbone=backbone)

    def test_missing_layer_name(self):
        with self.assertRaisesRegex(
            ValueError,
            "You have to specify the name of the low-level layer in the "
            "model used to extract low-level features.",
        ):
            backbone = models.DenseNet121(include_rescaling=True, include_top=False)
            segmentation.DeepLabV3Plus(
                classes=11,
                backbone=backbone,
                input_shape=(64, 64, 3),
            )

    def test_mixed_precision(self):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        backbone = models.ResNet50V2(
            include_rescaling=True,
            stackwise_dilations=[1, 1, 1, 2],
            include_top=False,
            input_shape=(256, 256, 3),
        )
        model = segmentation.DeepLabV3Plus(classes=11, backbone=backbone)
        input_image = tf.random.uniform(shape=[2, 256, 256, 3])
        output = model(input_image, training=True)

        self.assertEquals(output["output"].dtype, tf.float32)

    def test_invalid_backbone_model(self):
        with self.assertRaisesRegex(
            ValueError,
            "Backbone need to be a `tf.keras.layers.Layer`, received resnet_v3",
        ):
            segmentation.DeepLabV3Plus(
                classes=11,
                include_rescaling=True,
                stackwise_dilations=[1, 1, 1, 2],
                backbone="resnet_v3",
                low_level_feature_layer="lay1",
            )
        with self.assertRaisesRegex(
            ValueError, "Backbone need to be a `tf.keras.layers.Layer`"
        ):
            segmentation.DeepLabV3Plus(
                classes=11,
                backbone=tf.Module(),
                feature_layers="lay1",
            )


if __name__ == "__main__":
    tf.test.main()
