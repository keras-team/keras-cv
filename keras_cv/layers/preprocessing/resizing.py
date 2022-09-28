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

from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
import keras_cv.utils
import tensorflow as tf
from keras_cv import bounding_box


class Resizing(BaseImageAugmentationLayer):
    """A preprocessing layer which resizes images.

    This layer resizes an image input to a target height and width. The input
    should be a 4D (batched) or 3D (unbatched) tensor in `"channels_last"`
    format.  Input pixel values can be of any range (e.g. `[0., 1.)` or `[0,
    255]`) and of interger or floating point dtype. By default, the layer will
    output floats.

    This layer can be called on tf.RaggedTensor batches of input images of
    distinct sizes, and will resize the outputs to dense tensors of uniform
    size.

    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

    Args:
      height: Integer, the height of the output shape.
      width: Integer, the width of the output shape.
      interpolation: String, the interpolation method. Defaults to `"bilinear"`.
        Supports `"bilinear"`, `"nearest"`, `"bicubic"`, `"area"`, `"lanczos3"`,
        `"lanczos5"`, `"gaussian"`, `"mitchellcubic"`.
      crop_to_aspect_ratio: If True, resize the images without aspect
        ratio distortion. When the original aspect ratio differs from the target
        aspect ratio, the output image will be cropped so as to return the
        largest possible window in the image (of size `(height, width)`) that
        matches the target aspect ratio. By default
        (`crop_to_aspect_ratio=False`), aspect ratio may not be preserved.
    """

    def __init__(
        self,
        height,
        width,
        interpolation="bilinear",
        crop_to_aspect_ratio=False,
        bounding_box_format=None,
        **kwargs,
    ):
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.crop_to_aspect_ratio = crop_to_aspect_ratio
        self._interpolation_method = utils.get_interpolation(interpolation)
        self.bounding_box_format = bounding_box_format
        if crop_to_aspect_ratio and bounding_box_format:
            # TODO(lukewood): support `bounding_box.smart_resize()`
            raise ValueError(
                "Resizing() does not support `crop_to_aspect_ratio=True` "
                "and `bounding_box_format` at the same time.  In order to resize with "
                "bounding boxes, please pass `crop_to_aspect_ratio=False`."
            )
        super().__init__(**kwargs)

    def _batch_augment(self, inputs):
        self._validate_inputs(inputs)
        images = inputs.get("images", None)
        bounding_boxes = inputs.get("bounding_boxes", None)

        if images is None:
            raise ValueError(
                "Resizing expects images as an input, weight as a raw input Tensor or "
                "as a dictionary: "
                '{"images": images, {"bounding_boxes": bounding_boxes}}.'
                f"Got: inputs = {inputs}"
            )
        if self.interpolation == "nearest":
            input_dtype = self.compute_dtype
        else:
            input_dtype = tf.float32

        if bounding_boxes is not None:
            if self.bounding_box_format is None:
                raise ValueError(
                    "Resizing requires `bounding_box_format` to be set "
                    "when augmenting bounding boxes, but `self.bounding_box_format=None`."
                )
            bounding_boxes = bounding_box.convert_format(
                bounding_boxes,
                source=self.bounding_box_format,
                target="rel_xyxy",
                images=images,
            )

        size = [self.height, self.width]
        if self.crop_to_aspect_ratio:

            def resize_to_aspect(x):
                if isinstance(x, tf.RaggedTensor):
                    x = x.to_tensor()
                return tf.keras.utils.smart_resize(
                    x, size=size, interpolation=self._interpolation_method
                )

            if isinstance(images, tf.RaggedTensor):
                size_as_shape = tf.TensorShape(size)
                shape = size_as_shape + images.shape[-1:]
                spec = tf.TensorSpec(shape, input_dtype)
                images = tf.map_fn(resize_to_aspect, images, fn_output_signature=spec)
            else:
                images = resize_to_aspect(inputs)
        else:
            images = tf.image.resize(
                images, size=size, method=self._interpolation_method
            )

        images = tf.cast(outputs, self.compute_dtype)
        inputs["images"] = images
        if bounding_boxes is not None:
            inputs["bounding_boxes"] = bounding_box.convert_format(
                bounding_boxes,
                target="rel_xyxy",
                source=self.bounding_box_format,
                images=images,
            )
        return inputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        input_shape[H_AXIS] = self.height
        input_shape[W_AXIS] = self.width
        return tf.TensorShape(input_shape)

    def get_config(self):
        config = {
            "height": self.height,
            "width": self.width,
            "interpolation": self.interpolation,
            "crop_to_aspect_ratio": self.crop_to_aspect_ratio,
            "bounding_box_format": self.bounding_box_format,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
