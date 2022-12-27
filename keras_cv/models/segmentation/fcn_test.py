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

import pytest
import tensorflow as tf
import tensorflow_datasets as tfds

from keras_cv import models
from keras_cv.models.segmentation import FCN


class FCNTest(tf.test.TestCase):
    def test_fcn_model_with_vgg16_backbone_construction_with_preconfigured_setting(
        self,
    ):
        model = FCN(
            classes=11,
            backbone="vgg16",
            model_architecture="fcn8s",
            input_shape=(64, 64, 3),
        )
        input_image = tf.random.uniform(shape=[2, 64, 64, 3])
        output = model(input_image)

        self.assertEquals(output.shape, [2, 64, 64, 11])

    def test_fcn_model_with_vgg19_backbone_construction_with_preconfigured_setting(
        self,
    ):
        model = FCN(
            classes=11,
            backbone="vgg19",
            model_architecture="fcn8s",
            input_shape=(64, 64, 3),
        )
        input_image = tf.random.uniform(shape=[2, 64, 64, 3])
        output = model(input_image)

        self.assertEquals(output.shape, [2, 64, 64, 11])

    def test_fcn_model_with_vgg16_components(self):
        backbone = models.VGG16(include_rescaling=True, include_top=False)
        print(type(backbone))
        model = FCN(classes=11, backbone=backbone, input_shape=(64, 64, 3))

        input_image = tf.random.uniform(shape=[2, 64, 64, 3])
        output = model(input_image)

        self.assertEquals(output.shape, [2, 64, 64, 11])

    def test_fcn_model_with_vgg19_components(self):
        backbone = models.VGG19(include_rescaling=False, include_top=False)
        print(type(backbone))
        model = FCN(classes=11, backbone=backbone, input_shape=(64, 64, 3))

        input_image = tf.random.uniform(shape=[2, 64, 64, 3])
        output = model(input_image)

        self.assertEquals(output.shape, [2, 64, 64, 11])

    def test_mixed_precision(self):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        model = FCN(
            classes=11,
            backbone="vgg16",
            model_architecture="fcn8s",
            input_shape=(64, 64, 3),
        )
        input_image = tf.random.uniform(shape=[2, 64, 64, 3])
        output = model(input_image)

        self.assertEquals(output.dtype, tf.float32)

    def test_invalid_backbone_model(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Chosen `backbone` argument is not a valid allowed backbone. Possible options are ['vgg16', 'vgg19']",
        ):
            FCN(
                classes=11,
                backbone="resnet",
                model_architecture="fcn8s",
                input_shape=(256, 256, 3),
            )
        with self.assertRaisesRegex(
            ValueError,
            r"Invalid argument for parameter `backbone`. Accepted values are ['vgg16', 'vgg19'] or a `tf.keras.models.Model` instance with only `tf.keras.layers.Conv2D`, 'tf.keras.layers.MaxPooling2D' or `tf.keras.layers.Dense` layers",
        ):
            FCN(classes=11, backbone=tf.Module(), model_architecture="fcn8s")
        with self.assertRaisesRegex(
            ValueError,
            r"Entered `backbone` argument has custom layers. Include a `tf.keras.models.Model` with `keras.layers.Conv2D` or `keras.layers.MaxPooling2D` layers only.",
        ):
            input_tensor = tf.keras.Input(shape=(256, 256, 3))
            output_tensor = tf.keras.layers.AveragePooling2D(input_tensor)
            backbone_model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
            FCN(
                classes=11,
                backbone=backbone_model,
            )

    def test_invalid_model_architecture(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Invalid argument for parameter `model_architecture`. Accepted values are ['fcn8s', 'fcn16s', 'fcn32s']",
        ):
            FCN(
                classes=11,
                backbone="vgg16",
                model_architecture="fcn10s",
                input_shape=(256, 256, 3),
            )

    @pytest.mark.skipif(
        "INTEGRATION" not in os.environ or os.environ["INTEGRATION"] != "true",
        reason="Takes a long time to run, only runs when INTEGRATION "
        "environment variable is set.  To run the test please run: \n"
        "`INTEGRATION=true pytest keras_cv/",
    )
    def test_model_train(self):
        model = FCN(classes=11, backbone="vgg16", model_architecture="fcn8s")

        gcs_data_pattern = "gs://caltech_birds2011_mask/0.1.1/*.tfrecord*"
        features = tfds.features.FeaturesDict(
            {
                "bbox": tfds.features.BBoxFeature(),
                "image": tfds.features.Image(shape=(None, None, 3), dtype=tf.uint8),
                "image/filename": tfds.features.Text(),
                "label": tfds.features.ClassLabel(num_classes=200),
                "label_name": tfds.features.Text(),
                "segmentation_mask": tfds.features.Image(
                    shape=(None, None, 1), dtype=tf.uint8
                ),
            }
        )

        filenames = tf.io.gfile.glob(gcs_data_pattern)
        AUTO = tf.data.AUTOTUNE
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
        ds = ds.with_options(ignore_order)
        ds = ds.map(features.deserialize_example, num_parallel_calls=AUTO)

        target_size = [384, 384]
        output_res = [96, 96]
        num_images = 11788

        image_resizing = tf.keras.layers.Resizing(target_size[1], target_size[0])
        labels_resizing = tf.keras.layers.Resizing(output_res[1], output_res[0])

        def resize_images_and_masks(data):
            image = tf.image.convert_image_dtype(data["image"], dtype=tf.float32)
            data["image"] = image_resizing(image)
            # WARNING: assumes processing unbatched
            mask = data["segmentation_mask"]
            mask = tf.image.convert_image_dtype(mask, dtype=tf.float32)
            data["segmentation_mask"] = labels_resizing(mask)
            return data

        def keep_image_and_mask_only(data):
            return data["image"], data["segmentation_mask"]

        dataset = ds
        dataset = dataset.map(resize_images_and_masks)
        dataset = dataset.map(keep_image_and_mask_only)

        batch_size = 32
        training_dataset = dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(batch_size)
        )
        training_dataset = training_dataset.repeat()

        epochs = 1
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        model_history = model.fit(
            training_dataset, epochs=epochs, steps_per_epoch=num_images // batch_size
        )
        print(model_history)


if __name__ == "__main__":
    tf.test.main()
