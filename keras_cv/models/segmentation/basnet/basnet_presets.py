from keras_cv.models.backbones.resnet_v1 import resnet_v1_backbone_presets

basnet_presets = {
    "basnet_resnet34": {
        "metadata": {
            "description": (
                "BASNet with a ResNet34 v1 backbone. "
            ),
            "params": 108869058,
            "official_name": "BASNet",
            "path": "basnet",
        },
        "config": {
            "backbone": resnet_v1_backbone_presets.backbone_presets[
                "resnet34"
            ],
            "num_classes": 1,
            "input_shape": (288, 288, 3)
        },
    },
}
