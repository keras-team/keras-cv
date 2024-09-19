# Copyright 2024 The KerasCV Authors
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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras


@keras_cv_export(
    "keras_cv.models.mask_rcnn.MaskHead",
    package="keras_cv.models.mask_rcnn",
)
class MaskHead(keras.layers.Layer):
    """A Keras layer implementing the R-CNN Mask Head.

    The architecture is adopted from Matterport's Mask R-CNN implementation
    https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py.

    Args:
        num_classes: The number of object classes that are being detected,
            excluding the background class.
        stackwise_num_conv_filters: (Optional) a list of integers specifying
            the number of filters for each convolutional layer. Defaults
            to [256, 256].
        num_deconv_filters: (Optional) the number of filters to use in the
            upsampling convolutional layer. Defaults to 256.
    """

    def __init__(
        self,
        num_classes,
        stackwise_num_conv_filters=[256, 256],
        num_deconv_filters=256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.stackwise_num_conv_filters = stackwise_num_conv_filters
        self.num_deconv_filters = num_deconv_filters
        self.layers = []
        for num_filters in stackwise_num_conv_filters:
            conv = keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=3,
                padding="same",
            )
            batchnorm = keras.layers.BatchNormalization()
            activation = keras.layers.Activation("relu")
            self.layers.extend([conv, batchnorm, activation])

        self.deconv = keras.layers.Conv2DTranspose(
            num_deconv_filters,
            kernel_size=2,
            strides=2,
            activation="relu",
            padding="valid",
        )
        # we do not use a final sigmoid activation, since we use
        # from_logits=True during training
        self.segmentation_mask_output = keras.layers.Conv2D(
            num_classes + 1,
            kernel_size=1,
            strides=1,
            activation="linear",
        )

    def call(self, feature_map, training=False):
        # reshape batch and ROI axes into one axis to obtain a suitable
        # shape for conv layers
        num_rois = keras.ops.shape(feature_map)[1]
        x = keras.ops.reshape(feature_map, (-1, *feature_map.shape[2:]))
        for layer in self.layers:
            x = layer(x, training=training)
        x = self.deconv(x)
        segmentation_mask = self.segmentation_mask_output(x)
        segmentation_mask = keras.ops.reshape(
            segmentation_mask, (-1, num_rois, *segmentation_mask.shape[1:])
        )
        return segmentation_mask

    def build(self, input_shape):
        if input_shape[0] is None or input_shape[1] is None:
            intermediate_shape = (None, *input_shape[2:])
        else:
            intermediate_shape = (
                input_shape[0] * input_shape[1],
                *input_shape[2:],
            )
        for idx, num_filters in enumerate(self.stackwise_num_conv_filters):
            self.layers[idx * 3].build(intermediate_shape)
            intermediate_shape = tuple(intermediate_shape[:-1]) + (num_filters,)
            self.layers[idx * 3 + 1].build(intermediate_shape)
        self.deconv.build(intermediate_shape)
        intermediate_shape = tuple(intermediate_shape[:-3]) + (
            intermediate_shape[-3] * 2,
            intermediate_shape[-2] * 2,
            self.num_deconv_filters,
        )
        self.segmentation_mask_output.build(intermediate_shape)
        self.built = True

    def get_config(self):
        config = super().get_config()
        config["num_classes"] = self.num_classes
        config["stackwise_num_conv_filters"] = self.stackwise_num_conv_filters
        config["num_deconv_filters"] = self.num_deconv_filters
        return config
