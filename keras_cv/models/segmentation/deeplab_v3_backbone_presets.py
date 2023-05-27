from keras_cv.models import ResNet50V2Backbone

backbone_presets_no_weights = {
    "deeplab_v3_test": {
        "metadata": {
            "description": (
                "DeepLabV3 model with with ResNet50V2Backbone as feature "
                "extractor."
            ),
            "params": 39102976,
            "official_name": "DeepLabV3",
            "path": "deeplabv3"
        },
        "class_name": "keras_cv.models>DeepLabV3Backbone",
        "config": {
            "feature_extractor": ResNet50V2Backbone(input_shape=[64, 64, 3]),
            "spatial_pyramid_pooling": None,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    }
}
backbone_presets_with_weights = {
    "deeplab_v3_test_weights": {
        "metadata": {
            "description": (
                "DeepLabV3 model with with ResNet50V2Backbone as feature "
                "extractor."
            ),
            "params": 39102976,
            "official_name": "DeepLabV3",
            "path": "deeplabv3"
        },
        "class_name": "keras_cv.models>DeepLabV3Backbone",
        "config": backbone_presets_no_weights["deeplab_v3_test"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/mobilenetv3/mobilenetv3_large_imagenet_backbone.h5",   # noqa: E501
        "weights_hash": "ec55ea2f4f4ee9a2ddf3ee8e2dd784e9d5732690c1fc5afc7e1b2a66703f3337"  # noqa: E501
    }
}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
