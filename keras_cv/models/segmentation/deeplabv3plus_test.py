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
    def test_deeplab_model_construction_with_preconfigured_setting(self):
        model = segmentation.DeepLabV3Plus(
            classes=11,
            include_rescaling=True,
            backbone="resnet101_v2",
            input_shape=(256, 256, 3),
        )
        input_image = tf.random.uniform(shape=[2, 256, 256, 3])
        output = model(input_image, training=True)

        self.assertEquals(output["output"].shape, [2, 256, 256, 11])

    def test_deeplab_model_with_components(self):
        backbone = models.ResNet101V2(include_rescaling=True, include_top=False)
        model = segmentation.DeepLabV3Plus(
            classes=11,
            include_rescaling=True,
            backbone=backbone,
            feature_layers=("v2_stack_1_block4_1_relu", "v2_stack_3_block2_2_relu"),
            input_shape=(256, 256, 3),
        )

        input_image = tf.random.uniform(shape=[2, 256, 256, 3])
        output = model(input_image, training=True)

        self.assertEquals(output["output"].shape, [2, 256, 256, 11])

    def test_mixed_precision(self):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        model = segmentation.DeepLabV3Plus(
            classes=11,
            include_rescaling=True,
            backbone="resnet101_v2",
            input_shape=(256, 256, 3),
        )
        input_image = tf.random.uniform(shape=[2, 256, 256, 3])
        output = model(input_image, training=True)

        self.assertEquals(output["output"].dtype, tf.float32)

    def test_invalid_backbone_model(self):
        with self.assertRaisesRegex(
            ValueError, "Supported premade backbones are: .*resnet101_v2"
        ):
            segmentation.DeepLabV3(
                classes=11,
                include_rescaling=True,
                backbone="resnet_v3",
                feature_layers=("lay1", "lay2"),
            )
        with self.assertRaisesRegex(
            ValueError, "Backbone need to be a `tf.keras.layers.Layer`"
        ):
            segmentation.DeepLabV3Plus(
                classes=11,
                include_rescaling=True,
                backbone=tf.Module(),
                feature_layers=("lay1", "lay2"),
            )


if __name__ == "__main__":
    tf.test.main()
