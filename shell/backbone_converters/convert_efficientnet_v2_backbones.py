import hashlib
import json

import keras
import numpy as np
import tensorflow as tf

import keras_cv

filepath = tf.keras.utils.get_file(origin="https://i.imgur.com/9i63gLN.jpg")
image = keras.utils.load_img(filepath)
image = np.array(image)
image = np.array([image]).astype(float)

original_models_with_weights = [
    keras_cv.models.efficientnet_v2.EfficientNetV2S,
    keras_cv.models.efficientnet_v2.EfficientNetV2B0,
    keras_cv.models.efficientnet_v2.EfficientNetV2B1,
    keras_cv.models.efficientnet_v2.EfficientNetV2B2,
]
presets_with_weights = [
    "efficientnetv2_s_imagenet_classifier",
    "efficientnetv2_b0_imagenet_classifier",
    "efficientnetv2_b1_imagenet_classifier",
    "efficientnetv2_b2_imagenet_classifier",
]

preset_updates = {}

for original_model_cls, preset_name in zip(
    original_models_with_weights, presets_with_weights
):
    original_model = original_model_cls(
        include_rescaling=True,
        include_top=True,
        num_classes=1000,
        weights="imagenet",
    )

    model = keras_cv.models.ImageClassifier.from_preset(
        preset_name, load_weights=False
    )

    original_layers = list(original_model._flatten_layers())
    original_layers = [
        layer for layer in original_layers if "dropout" not in layer.name
    ]

    new_layers = list(model._flatten_layers())
    new_layers = [layer for layer in new_layers if "backbone" not in layer.name]

    for original_layer, new_layer in zip(original_layers, new_layers):
        new_layer.set_weights(original_layer.get_weights())

    output_one = model.predict(image)
    output_two = original_model.predict(image)
    deltas = output_one - output_two

    # As tiny delta as possible
    delta = 0.00001
    assert all(((output_one - output_two) < delta).flatten().tolist())

    weights_path = f"efficientnet_v2/{preset_name}.h5"
    model.save_weights(weights_path)
    weights_hash = hashlib.md5(open(weights_path, "rb").read()).hexdigest()

    preset_updates[preset_name] = {
        "weights_url": f"https://storage.googleapis.com/keras-cv/models/{weights_path}",  # noqa: E501
        "weights_hash": weights_hash,
    }

with open("efficientnet_v2/preset_updates.json", "w") as f:
    json.dump(preset_updates, f, indent=4)

print("Please run:")
print("`gsutil cp -r efficientnet_v2/ gs://keras-cv/models/`")
print('`gsutil acl ch -u AllUsers:R "gs://keras-cv/models/efficientnet_v2/*"`')
