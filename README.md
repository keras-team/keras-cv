# KerasCV

[![](https://github.com/keras-team/keras-cv/workflows/Tests/badge.svg?branch=master)](https://github.com/keras-team/keras-cv/actions?query=workflow%3ATests+branch%3Amaster)
![Downloads](https://img.shields.io/pypi/dm/keras-cv.svg)
![Python](https://img.shields.io/badge/python-v3.7.0+-success.svg)
![Tensorflow](https://img.shields.io/badge/tensorflow-v2.9.0+-success.svg)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/keras-team/keras-cv/issues)

KerasCV is a library of modular computer vision components that work natively
with TensorFlow, JAX, or PyTorch. Built on [Keras Core](https://keras.io/keras_core/announcement/),
these models, layers, metrics, callbacks, etc., can be trained and serialized
in any framework and re-used in another without costly migrations. See
"Configuring your backend" below for more details on multi-framework KerasCV.

<img style="width: 440px; max-width: 90%;" src="https://storage.googleapis.com/keras-cv/guides/keras-cv-augmentations.gif">

KerasCV can be understood as a horizontal extension of the Keras API: the
components are new first-party Keras objects that are too specialized to be
added to core Keras. They receive the same level of polish and backwards
compatibility guarantees as the core Keras API, and they are maintained by the
Keras team.

Our APIs assist in common computer vision tasks such as data augmentation,
classification, object detection, segmentation, image generation, and more.
Applied computer vision engineers can leverage KerasCV to quickly assemble
production-grade, state-of-the-art training and inference pipelines for all of
these common tasks.

## Quick Links
- [List of available models and presets](https://keras.io/api/keras_cv/models/)
- [Developer Guides](https://keras.io/guides/keras_cv/)
- [Contributing Guide](.github/CONTRIBUTING.md)
- [Call for Contributions](https://github.com/keras-team/keras-cv/issues?q=is%3Aopen+is%3Aissue+label%3Acontribution-welcome)
- [API Design Guidelines](.github/API_DESIGN.md)

## Installation

To install the latest official release:

```
pip install keras-cv tensorflow --upgrade
```

To install the latest unreleased changes to the library, we recommend using
pip to install directly from the master branch on github:

```
pip install git+https://github.com/keras-team/keras-cv.git tensorflow --upgrade
```

## Configuring your backend

**Keras 3** is an upcoming release of the Keras library which supports
TensorFlow, Jax or Torch as backends. This is supported today in KerasNLP,
but will not be enabled by default until the official release of Keras 3. If you
`pip install keras-cv` and run a script or notebook without changes, you will
be using TensorFlow and **Keras 2**.

If you would like to enable a preview of the Keras 3 behavior, you can do
so by setting the `KERAS_BACKEND` environment variable. For example:

```shell
export KERAS_BACKEND=jax
```

Or in Colab, with:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras_cv
```

> [!IMPORTANT]
> Make sure to set the `KERAS_BACKEND` before import any Keras libraries, it
> will be used to set up Keras when it is first imported.
Until the Keras 3 release, KerasCV will use a preview of Keras 3 on PyPI named
[keras-core](https://pypi.org/project/keras-core/).

> [!IMPORTANT]
> If you set `KERAS_BACKEND` variable, you should `import keras_core as keras`
> instead of `import keras`. This is a temporary step until Keras 3 is out!
To restore the default **Keras 2** behavior, `unset KERAS_BACKEND` before
importing Keras and KerasCV.

Once that configuration step is done, you can just import KerasCV and start
using it on top of your backend of choice:

```python
import keras_cv
from keras_cv.backend import keras

filepath = keras.utils.get_file(origin="https://i.imgur.com/gCNcJJI.jpg")
image = np.array(keras.utils.load_img(filepath))
image_resized = keras.ops.image.resize(image, (640, 640))[None, ...]

model = keras_cv.models.YOLOV8Detector.from_preset(
    "yolo_v8_m_pascalvoc",
    bounding_box_format="xywh",
)
predictions = model.predict(image_resized)
```

## Quickstart

```python
import tensorflow as tf
import keras_cv
import tensorflow_datasets as tfds
from keras_cv.backend import keras

# Create a preprocessing pipeline with augmentations
BATCH_SIZE = 16
NUM_CLASSES = 3
augmenter = keras_cv.layers.Augmenter(
    [
        keras_cv.layers.RandomFlip(),
        keras_cv.layers.RandAugment(value_range=(0, 255)),
        keras_cv.layers.CutMix(),
    ],
)

def preprocess_data(images, labels, augment=False):
    labels = tf.one_hot(labels, NUM_CLASSES)
    inputs = {"images": images, "labels": labels}
    outputs = inputs
    if augment:
        outputs = augmenter(outputs)
    return outputs['images'], outputs['labels']

train_dataset, test_dataset = tfds.load(
    'rock_paper_scissors',
    as_supervised=True,
    split=['train', 'test'],
)
train_dataset = train_dataset.batch(BATCH_SIZE).map(
    lambda x, y: preprocess_data(x, y, augment=True),
        num_parallel_calls=tf.data.AUTOTUNE).prefetch(
            tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).map(
    preprocess_data, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
        tf.data.AUTOTUNE)

# Create a model using a pretrained backbone
backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(
    "efficientnetv2_b0_imagenet"
)
model = keras_cv.models.ImageClassifier(
    backbone=backbone,
    num_classes=NUM_CLASSES,
    activation="softmax",
)
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    metrics=['accuracy']
)

# Train your model
model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=8,
)
```

## Contributors
If you'd like to contribute, please see our [contributing guide](.github/CONTRIBUTING.md).

To find an issue to tackle, please check our [call for contributions](.github/CALL_FOR_CONTRIBUTIONS.md).

We would like to leverage/outsource the Keras community not only for bug reporting,
but also for active development for feature delivery. To achieve this, here is the predefined
process for how to contribute to this repository:

1) Contributors are always welcome to help us fix an issue, add tests, better documentation.
2) If contributors would like to create a backbone, we usually require a pre-trained weight set
with the model for one dataset as the first PR, and a training script as a follow-up. The training script will preferably help us reproduce the results claimed from paper. The backbone should be generic but the training script can contain paper specific parameters such as learning rate schedules and weight decays. The training script will be used to produce leaderboard results.
Exceptions apply to large transformer-based models which are difficult to train. If this is the case,
contributors should let us know so the team can help in training the model or providing GCP resources.
3) If contributors would like to create a meta arch, please try to be aligned with our roadmap and create a PR for design review to make sure the meta arch is modular.
4) If contributors would like to create a new input formatting which is not in our roadmap for the next 6 months, e.g., keypoint, please create an issue and ask for a sponsor.
5) If contributors would like to support a new task which is not in our roadmap for the next 6 months, e.g., 3D reconstruction, please create an issue and ask for a sponsor.

Thank you to all of our wonderful contributors!

<a href="https://github.com/keras-team/keras-cv/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=keras-team/keras-cv" />
</a>

## Pretrained Weights
Many models in KerasCV come with pre-trained weights.
With the exception of StableDiffusion and the standard Vision Transformer, all of these weights are trained using Keras and
KerasCV components and training scripts in this repository.
While some models are not trained with the same parameters or preprocessing pipeline
as defined in their original publications, the KerasCV team ensures strong numerical performance.
Performance metrics for the provided pre-trained weights can be found
in the training history for each documented task.
An example of this can be found in the ImageNet classification training
[history for backbone models](examples/training/classification/imagenet/training_history.json).
All results are reproducible using the training scripts in this repository.

Historically, many models have been trained on image datasets rescaled via manually
crafted normalization schemes.
The most common variant of manually crafted normalization scheme is subtraction of the
imagenet mean pixel followed by standard deviation normalization based on the imagenet
pixel standard deviation.
This scheme is an artifact of the days of manual feature engineering, but is no longer
required to score state of the art scores using modern deep learning architectures.
Due to this, KerasCV is standardized to operate on images that have been rescaled using
a simple `1/255` rescaling layer.
This can be seen in all KerasCV training pipelines and code examples.

## Custom Ops
Note that in some of the 3D Object Detection layers, custom TF ops are used. The
binaries for these ops are not shipped in our PyPi package in order to keep our
wheels pure-Python.

If you'd like to use these custom ops, you can install from source using the
instructions below.

### Installing KerasCV with Custom Ops from Source

Installing custom ops from source requires the [Bazel](https://bazel.build/) build
system (version >= 5.4.0). Steps to install Bazel can be [found here](https://github.com/keras-team/keras/blob/v2.11.0/.devcontainer/Dockerfile#L21-L23).

```
git clone https://github.com/keras-team/keras-cv.git
cd keras-cv

python3 build_deps/configure.py

bazel build build_pip_pkg
export BUILD_WITH_CUSTOM_OPS=true
bazel-bin/build_pip_pkg wheels

pip install wheels/keras_cv-*.whl
```

Note that GitHub actions exist to release KerasCV with custom ops, but are
currently disabled. You can use these [actions](https://github.com/keras-team/keras-cv/blob/master/.github/workflows/release.yml)
in your own fork to create wheels for Linux (manylinux2014), MacOS (both x86 and ARM),
and Windows.

## Disclaimer

KerasCV provides access to pre-trained models via the `keras_cv.models` API.
These pre-trained models are provided on an "as is" basis, without warranties
or conditions of any kind.
The following underlying models are provided by third parties, and are subject to separate
licenses:
StableDiffusion, Vision Transformer

## Citing KerasCV

If KerasCV helps your research, we appreciate your citations.
Here is the BibTeX entry:

```bibtex
@misc{wood2022kerascv,
  title={KerasCV},
  author={Wood, Luke and Tan, Zhenyu and Stenbit, Ian and Bischof, Jonathan and Zhu, Scott and Chollet, Fran\c{c}ois and others},
  year={2022},
  howpublished={\url{https://github.com/keras-team/keras-cv}},
}
```
