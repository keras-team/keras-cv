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

import keras_cv.utils
from keras_cv import bounding_box
from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)

H_AXIS = -3
W_AXIS = -2


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
        pad_to_aspect_ratio=False,
        bounding_box_format=None,
        **kwargs,
    ):
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.crop_to_aspect_ratio = crop_to_aspect_ratio
        self.pad_to_aspect_ratio = pad_to_aspect_ratio
        self._interpolation_method = keras_cv.utils.get_interpolation(interpolation)
        self.bounding_box_format = bounding_box_format

        if pad_to_aspect_ratio and crop_to_aspect_ratio:
            raise ValueError(
                "`Resizing()` expects either `crop_to_aspect_ratio` or "
                "`pad_to_aspect_ratio`, but not both."
            )
        if crop_to_aspect_ratio and bounding_box_format:
            # TODO(lukewood): support `bounding_box.smart_resize()`
            raise ValueError(
                "Resizing() does not support `crop_to_aspect_ratio=True` "
                "and `bounding_box_format` at the same time.  In order to resize with "
                "bounding boxes, please pass `crop_to_aspect_ratio=False`."
            )
        super().__init__(**kwargs)

    def _augment(self, inputs):
        images = inputs.get("images", None)
        bounding_boxes = inputs.get("bounding_boxes", None)

        if images is not None:
            images = tf.expand_dims(images, axis=0)
        if bounding_boxes is not None:
            bounding_boxes = tf.expand_dims(bounding_boxes, axis=0)

        inputs["images"] = images
        inputs["bounding_boxes"] = bounding_boxes

        inputs = self._batch_augment(inputs)

        if images is not None:
            images = tf.squeeze(images, axis=0)
        if bounding_boxes is not None:
            bounding_boxes = tf.squeeze(bounding_boxes, axis=0)

        inputs["images"] = images
        inputs["bounding_boxes"] = bounding_boxes
        return inputs

    def _resize_with_distortion(self, inputs):
        images = inputs.get("images", None)
        bounding_boxes = inputs.get("bounding_boxes", None)
        if bounding_boxes is not None:
            raise ValueError(
                "Resizing() only supports bounding box inputs when "
                "`pad_to_aspect_ratio=True`.  Please construct your layer with "
                "`Resizing(pad_to_aspect_ratio=True)` when processing "
                "bounding boxes."
            )

        size = [self.height, self.width]
        images = tf.image.resize(images, size=size, method=self._interpolation_method)
        images = tf.cast(images, self.compute_dtype)

        inputs["images"] = images
        return inputs

    def _resize_with_pad(self, inputs):
        def resize_single_with_pad_to_aspect(x):
            image = x.get("images", None)
            boxes = x.get("bounding_boxes", None)
            # images must be dense-able at this point.
            image = image.to_tensor()
            img_size = tf.shape(image)
            img_height = tf.cast(img_size[H_AXIS], self.compute_dtype)
            img_width = tf.cast(img_size[W_AXIS], self.compute_dtype)

            if boxes is not None:
                boxes = keras_cv.bounding_box.convert_format(
                    boxes,
                    image_shape=img_size,
                    source=self.bounding_box_format,
                    target="rel_xyxy",
                )

            # how much we scale height by to hit target height
            height_scale = self.height / img_height
            width_scale = self.width / img_width

            resize_scale = tf.math.minimum(height_scale, width_scale)
            target_height = img_height * resize_scale
            target_width = img_width * resize_scale

            image = tf.image.resize(
                image,
                size=(target_height, target_width),
                method=self._interpolation_method,
            )
            if boxes is not None:
                boxes = keras_cv.bounding_box.convert_format(
                    boxes,
                    images=image,
                    source="rel_xyxy",
                    target="xyxy",
                )
            image = tf.image.pad_to_bounding_box(image, 0, 0, self.height, self.width)
            if boxes is not None:
                boxes = keras_cv.bounding_box.convert_format(
                    boxes,
                    images=image,
                    source="xyxy",
                    target=self.bounding_box_format,
                )
            inputs["images"] = image
            if boxes is not None:
                inputs["bounding_boxes"] = tf.RaggedTensor.from_tensor(boxes)
            return inputs

        size_as_shape = tf.TensorShape((self.height, self.width))
        shape = size_as_shape + inputs["images"].shape[-1:]
        img_spec = tf.TensorSpec(shape, self.compute_dtype)
        boxes_spec = tf.RaggedTensorSpec(
            shape=[None, inputs["bounding_boxes"].shape[2]],
            dtype=inputs["bounding_boxes"].dtype,
        )
        return tf.map_fn(
            resize_single_with_pad_to_aspect,
            inputs,
            fn_output_signature={"images": img_spec, "bounding_boxes": boxes_spec},
        )

    def _resize_with_crop(self, inputs):
        images = inputs.get("images", None)
        bounding_boxes = inputs.get("bounding_boxes", None)
        if bounding_boxes is not None:
            raise ValueError(
                "Resizing(crop_to_aspect_ratio=True) does not support "
                "bounding box inputs.  Please use `pad_to_aspect_ratio=True`, or "
                "`crop_to_aspect_ratio=False` and `pad_to_aspect_ratio=False` when "
                "passing bounding boxes to Resizing()."
            )
        inputs["images"] = images
        size = [self.height, self.width]

        def resize_with_crop_to_aspect(x):
            if isinstance(x, tf.RaggedTensor):
                x = x.to_tensor()
            return tf.keras.utils.smart_resize(
                x, size=size, interpolation=self._interpolation_method
            )

        if isinstance(images, tf.RaggedTensor):
            size_as_shape = tf.TensorShape(size)
            shape = size_as_shape + images.shape[-1:]
            spec = tf.TensorSpec(shape, input_dtype)
            images = tf.map_fn(
                resize_with_crop_to_aspect, images, fn_output_signature=spec
            )
        else:
            images = resize_with_crop_to_aspect(images)

        inputs["images"] = images
        return inputs

    def call(self, inputs, training=True):
        # inputs = self._ensure_inputs_are_compute_dtype(inputs)
        inputs, metadata = self._format_inputs(inputs)
        images = inputs["images"]
        if images.shape.rank == 3:
            return self._format_output(self._augment(inputs), metadata)
        elif images.shape.rank == 4:
            return self._format_output(self._batch_augment(inputs), metadata)
        else:
            raise ValueError(
                "Image augmentation layers are expecting inputs to be "
                "rank 3 (HWC) or 4D (NHWC) tensors. Got shape: "
                f"{images.shape}"
            )

    def _batch_augment(self, inputs):
        if (
            inputs.get("bounding_boxes", None) is None
            and self.bounding_box_format is None
        ):
            raise ValueError(
                "Resizing requires `bounding_box_format` to be set "
                "when augmenting bounding boxes, but `self.bounding_box_format=None`."
            )

        if self.crop_to_aspect_ratio:
            return self._resize_with_crop(inputs)
        if self.pad_to_aspect_ratio:
            return self._resize_with_pad(inputs)
        return self._resize_with_distortion(inputs)

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
            "pad_to_aspect_ratio": self.pad_to_aspect_ratio,
            "bounding_box_format": self.bounding_box_format,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
