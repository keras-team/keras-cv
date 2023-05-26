backbone_presets_no_weights = {
    "deeplab_v3_test": {
        "metadata": {},
        "class_name": "keras_cv.models>DeepLabV3Backbone",
        "config": {},
    }
}
backbone_presets_with_weights = {}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
