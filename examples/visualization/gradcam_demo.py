import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras_cv import visualization
from keras_cv.visualization.utils import normalize

SIZES = (299, 299)
LABELS = 5

# Load images from the web
paths = [
    tf.keras.utils.get_file("african_elephant.jpg", "https://i.imgur.com/Bvro0YD.png"),
    tf.keras.utils.get_file("wolf.jpg", "https://i.imgur.com/rRcsL2D.jpeg"),
    tf.keras.utils.get_file("cat_and_dog.jpg", "https://i.imgur.com/6wxtMEp.jpeg"),
]


def load_and_prepare(paths):
    return np.stack(
        [
            tf.keras.preprocessing.image.img_to_array(
                tf.keras.preprocessing.image.load_img(path, target_size=SIZES)
            )
            for path in paths
        ]
    ).astype("uint8")


images = load_and_prepare(paths)
x = tf.convert_to_tensor(images.astype("float32") / 127.5 - 1)

# Load pre-trained model over ImageNet
input_tensor = tf.keras.Input([None, None, 3], name="images")
rn50 = tf.keras.applications.ResNet50V2(
    input_tensor=input_tensor, weights="imagenet", classifier_activation=None
)

# Collect the input tensor of the average pooling layer,
# the most common positional signal used when computing Grad-CAM:
endpoints = visualization.utils.collect_endpoints(
    rn50,
    endpoints=[
        {"name": "avg_pool", "link": "input"},
        {"name": "predictions"},
    ],
)

rn50_w_acts = tf.keras.Model(
    inputs=rn50.inputs,
    outputs=endpoints,
)

# For each image, get the index of the 5 labels with
# highest associated classification energy:
logits = rn50.predict(x, verbose=0)
labels = tf.argsort(logits, axis=-1, direction="DESCENDING")[..., :LABELS]
probs = tf.nn.softmax(logits)
decoded = tf.keras.applications.imagenet_utils.decode_predictions(probs.numpy())

logits, maps = visualization.gradcam(rn50_w_acts, x, labels)

# # We are not concerned with pixels that negatively contribute
# # to its classification, only pixels that belong to that class.
maps = tf.nn.relu(maps)
maps = normalize(maps)
maps = tf.image.resize(maps, SIZES).numpy()

rows = len(images)
cols = LABELS

plt.figure(figsize=(4 * rows, 4 * cols))
for sid, (s_preds, s_maps) in enumerate(zip(decoded, maps)):
    for label in range(LABELS):
        _id, name, confidence = s_preds[label]
        plt.subplot(rows, cols, sid * cols + label + 1)
        plt.title(f"{name} {confidence:.2%}")
        plt.imshow(images[sid])
        plt.imshow(s_maps[..., label], cmap="jet", alpha=0.5)
        plt.axis("off")
plt.show()
