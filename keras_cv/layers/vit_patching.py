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
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class Patching(layers.Layer):
    """
    Layer to turn images into a sequence of patches for Vision Transformers from:
        - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
        by Alexey Dosovitskiy et al. (https://arxiv.org/abs/2010.11929)
    Based on Khalid Salama's implementation for:
        - https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_with_vision_transformer.py

    The layer expects a batch of input images and returns batches of patches.

    args:
        - patch_size: the size (patch_size, patch_size) of each patch created from the image

    Basic usage:

    ```
    img = url_to_array(url)
    batch_img = np.expand_dims(img, 0)

    patch_size = 16
    patches = keras_cv.layers.Patching(patch_size)(batch_img)
    print(f"Image size: {batch_img.shape}")                # Image size: (1, 224, 224, 3)
    print(f"Patch size: {(patch_size, patch_size)}") # Patch size: (16, 16)
    print(f"Patches: {patches.shape[1]}")            # Patches: 196
    ```
    """

    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = {
            "patch_size": self.patch_size,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
