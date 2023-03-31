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

"""CSPDarkNet model preset configurations."""

backbone_presets_no_weights = {
    "cspdarknettiny": {
        "metadata": {
            "description": (
                "CSPDarkNet model with 0.33 depth multiplier and 0.375 width "
                "multiplier where the batch normalization and SiLU activation "
                "are applied after the convolution layers."
            ),
        },
        "class_name": "keras_cv.models>CSPDarkNetBackbone",
        "config": {
            "depth_multiplier": 0.33,
            "width_multiplier": 0.375,
            "include_rescaling": True,
            "use_depthwise": False,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "cspdarknets": {
        "metadata": {
            "description": (
                "CSPDarkNet model with 0.33 depth multiplier and 0.50 width "
                "multiplier where the batch normalization and SiLU activation "
                "are applied after the convolution layers."
            ),
        },
        "class_name": "keras_cv.models>CSPDarkNetBackbone",
        "config": {
            "depth_multiplier": 0.33,
            "width_multiplier": 0.50,
            "include_rescaling": True,
            "use_depthwise": False,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "cspdarknetm": {
        "metadata": {
            "description": (
                "CSPDarkNet model with 0.67 depth multiplier and 0.75 width "
                "multiplier where the batch normalization and SiLU activation "
                "are applied after the convolution layers."
            ),
        },
        "class_name": "keras_cv.models>CSPDarkNetBackbone",
        "config": {
            "depth_multiplier": 0.67,
            "width_multiplier": 0.75,
            "include_rescaling": True,
            "use_depthwise": False,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "cspdarknetl": {
        "metadata": {
            "description": (
                "CSPDarkNet model with 1.00 depth multiplier and 1.00 width "
                "multiplier where the batch normalization and SiLU activation "
                "are applied after the convolution layers."
            ),
        },
        "class_name": "keras_cv.models>CSPDarkNetBackbone",
        "config": {
            "depth_multiplier": 1.00,
            "width_multiplier": 1.00,
            "include_rescaling": True,
            "use_depthwise": False,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "cspdarknetx": {
        "metadata": {
            "description": (
                "CSPDarkNet model with 1.33 depth multiplier and 1.25 width "
                "multiplier where the batch normalization and SiLU activation "
                "are applied after the convolution layers."
            ),
        },
        "class_name": "keras_cv.models>CSPDarkNetBackbone",
        "config": {
            "depth_multiplier": 1.33,
            "width_multiplier": 1.25,
            "include_rescaling": True,
            "use_depthwise": False,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
}

backbone_presets_with_weights = {
    "cspdarknettiny_imagenet": {
        "metadata": {
            "description": (
                "CSPDarkNet model with 0.33 depth multiplier and 0.375 width "
                "multiplier where the batch normalization and SiLU activation "
                "are applied after the convolution layers. Trained on Imagenet "
                "2012 classification task."
            ),
        },
        "class_name": "keras_cv.models>CSPDarkNetBackbone",
        "config": backbone_presets_no_weights["cspdarknettiny"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/cspdarknettiny/imagenet/classification-v0-notop.h5",
        "weights_hash": "0007ae82c95be4d4aef06368a7c38e006381324d77e5df029b04890e18a8ad19",
    },
    "cspdarknetl_imagenet": {
        "metadata": {
            "description": (
                "CSPDarkNet model with 1.00 depth multiplier and 1.00 width "
                "multiplier where the batch normalization and SiLU activation "
                "are applied after the convolution layers. Trained on Imagenet "
                "2012 classification task."
            ),
        },
        "class_name": "keras_cv.models>CSPDarkNetBackbone",
        "config": backbone_presets_no_weights["cspdarknetl"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/cspdarknetl/imagenet/classification-v0-notop.h5",
        "weights_hash": "9303aabfadffbff8447171fce1e941f96d230d8f3cef30d3f05a9c85097f8f1e",
    },
}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
