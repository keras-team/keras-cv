# Copyright 2023 The KerasCV Authors
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

"""EfficientNetV2 model preset configurations."""

DESCRIPTION = "One of the many EfficientNetV2 variants.  Each variant is built "
"based on one of the parameterizations described in the original EfficientNetV2"
"publication.  To learn more about the parameterizations and their tradeoffs, "
"please check keras.io.  As a starting point, we recommend starting with the "
'"efficientnetv2-s" architecture, and increasing in size to "efficientnetv2-m" '
'or "efficientnetv2-l" if resources permit.'

backbone_presets_no_weights = {
    "efficientnetv2-s": {
        "model_name": "efficientnetv2-s",
        "metadata": {
            "description": DESCRIPTION,
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "model_name": "efficientnetv2-s",
            "width_coefficient": 1.0,
            "depth_coefficient": 1.0,
            "default_size": 384,
            "dropout_rate": 0.2,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "bn_momentum": 0.9,
            "activation": "swish",
            "blocks_args": "default",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "efficientnetv2-m": {
        "metadata": {
            "description": DESCRIPTION,
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "model_name": "efficientnetv2-m",
            "width_coefficient": 1.0,
            "depth_coefficient": 1.0,
            "default_size": 480,
            "dropout_rate": 0.2,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "bn_momentum": 0.9,
            "activation": "swish",
            "blocks_args": "default",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "efficientnetv2-l": {
        "metadata": {
            "description": DESCRIPTION,
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "model_name": "efficientnetv2-l",
            "width_coefficient": 1.0,
            "depth_coefficient": 1.0,
            "default_size": 480,
            "dropout_rate": 0.2,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "bn_momentum": 0.9,
            "activation": "swish",
            "blocks_args": "default",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "efficientnetv2-b0": {
        "metadata": {
            "description": DESCRIPTION,
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "model_name": "efficientnetv2-b0",
            "width_coefficient": 1.0,
            "depth_coefficient": 1.0,
            "default_size": 224,
            "dropout_rate": 0.2,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "bn_momentum": 0.9,
            "activation": "swish",
            "blocks_args": "default",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "efficientnetv2-b1": {
        "metadata": {
            "description": DESCRIPTION,
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "model_name": "efficientnetv2-b1",
            "width_coefficient": 1.0,
            "depth_coefficient": 1.1,
            "default_size": 240,
            "dropout_rate": 0.2,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "bn_momentum": 0.9,
            "activation": "swish",
            "blocks_args": "default",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "efficientnetv2-b2": {
        "metadata": {
            "description": DESCRIPTION,
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "model_name": "efficientnetv2-b2",
            "width_coefficient": 1.1,
            "depth_coefficient": 1.2,
            "default_size": 260,
            "dropout_rate": 0.2,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "bn_momentum": 0.9,
            "activation": "swish",
            "blocks_args": "default",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "efficientnetv2-b3": {
        "metadata": {
            "description": DESCRIPTION,
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "model_name": "efficientnetv2-b3",
            "width_coefficient": 1.2,
            "depth_coefficient": 1.4,
            "default_size": 300,
            "dropout_rate": 0.2,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "bn_momentum": 0.9,
            "activation": "swish",
            "blocks_args": "default",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
}

backbone_presets_with_weights = {
    "efficientnetv2-s_imagenet": {
        "metadata": {
            "description": DESCRIPTION,
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "model_name": "efficientnetv2-s",
            "width_coefficient": 1.0,
            "depth_coefficient": 1.0,
            "default_size": 384,
            "dropout_rate": 0.2,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "bn_momentum": 0.9,
            "activation": "swish",
            "blocks_args": "default",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/efficientnetv2s/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "80555436ea49100893552614b4dce98de461fa3b6c14f8132673817d28c83654",  # noqa: E501
    },
    "efficientnetv2-b0_imagenet": {
        "metadata": {
            "description": "EfficientNetv2 model",
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "model_name": "efficientnetv2-b0",
            "width_coefficient": 1.0,
            "depth_coefficient": 1.0,
            "default_size": 224,
            "dropout_rate": 0.2,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "bn_momentum": 0.9,
            "activation": "swish",
            "blocks_args": "default",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/efficientnetv2b0/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "ac95f13a8ad1cee41184fc16fd0eb769f7c5b3131151c6abf7fcee5cc3d09bc8",  # noqa: E501
    },
    "efficientnetv2-b1_imagenet": {
        "metadata": {
            "description": DESCRIPTION,
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "model_name": "efficientnetv2-b1",
            "width_coefficient": 1.0,
            "depth_coefficient": 1.1,
            "default_size": 240,
            "dropout_rate": 0.2,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "bn_momentum": 0.9,
            "activation": "swish",
            "blocks_args": "default",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/efficientnetv2b1/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "82da111f8411f47e3f5eef090da76340f38e222f90a08bead53662f2ebafb01c",  # noqa: E501
    },
    "efficientnetv2-b2_imagenet": {
        "metadata": {
            "description": DESCRIPTION,
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "model_name": "efficientnetv2-b2",
            "width_coefficient": 1.1,
            "depth_coefficient": 1.2,
            "default_size": 260,
            "dropout_rate": 0.2,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "bn_momentum": 0.9,
            "activation": "swish",
            "blocks_args": "default",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/efficientnetv2b2/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "02d12c9d1589b540b4e84ffdb54ff30c96099bd59e311a85ddc7180efc65e955",  # noqa: E501
    },
}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
