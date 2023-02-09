# KerasCV

[![](https://github.com/keras-team/keras-cv/workflows/Tests/badge.svg?branch=master)](https://github.com/keras-team/keras-cv/actions?query=workflow%3ATests+branch%3Amaster)
![Downloads](https://img.shields.io/pypi/dm/keras-cv.svg)
![Python](https://img.shields.io/badge/python-v3.7.0+-success.svg)
![Tensorflow](https://img.shields.io/badge/tensorflow-v2.9.0+-success.svg)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/keras-team/keras-cv/issues)

KerasCV is a library of modular computer vision oriented Keras components.
These components include models, layers, metrics, losses, callbacks, and utility
functions.

KerasCV's primary goal is to provide a coherent, elegant, and pleasant API to train state of the art computer vision models.
Users should be able to train state of the art models using only `Keras`, `KerasCV`, and TensorFlow core (i.e. `tf.data`) components.

KerasCV can be understood as a horizontal extension of the Keras API: the components are new first-party
Keras objects (layers, metrics, etc.) that are too specialized to be added to core Keras. They receive the same level of polish and backwards compatibility guarantees as the core Keras API, and they are maintained by the Keras team.

Our APIs assist in common computer vision tasks such as data-augmentation, classification, object detection, image generation, and more.
Applied computer vision engineers can leverage KerasCV to quickly assemble production-grade, state-of-the-art training and inference pipelines for all of these common tasks.

In addition to API consistency, KerasCV components aim to be mixed-precision compatible, QAT compatible, XLA compilable, and TPU compatible.
We also aim to provide generic model optimization tools for deployment on devices such as onboard GPUs, mobile, and edge chips.


To learn more about the future project direction, please check the [roadmap](.github/ROADMAP.md).

## Quick Links
- [Contributing Guide](.github/CONTRIBUTING.md)
- [Call for Contributions](https://github.com/keras-team/keras-cv/issues?q=is%3Aopen+is%3Aissue+label%3Acontribution-welcome)
- [Roadmap](.github/ROADMAP.md)
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

## Quickstart

```python
# Create a preprocessing pipeline
import keras_cv
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

augmenter = keras_cv.layers.Augmenter(
    layers=[
        keras_cv.layers.RandomFlip(),
        keras_cv.layers.RandAugment(value_range=(0, 255)),
        keras_cv.layers.CutMix(),
        keras_cv.layers.MixUp()
    ]
)

def preprocess_data(images, labels, augment=False):
    labels = tf.one_hot(labels, 3)
    inputs = {"images": images, "labels": labels}
    outputs = augmenter(inputs) if augment else inputs
    return outputs['images'], outputs['labels']

# Augment a `tf.data.Dataset`
train_dataset, test_dataset = tfds.load(
    'rock_paper_scissors',
    as_supervised=True,
    split=['train', 'test'],
)
train_dataset = train_dataset.batch(16).map(
    lambda x, y: preprocess_data(x, y, augment=True),
        num_parallel_calls=tf.data.AUTOTUNE).prefetch(
            tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(16).map(
    preprocess_data, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
        tf.data.AUTOTUNE)

# Create a model
densenet = keras_cv.models.DenseNet121(
    include_rescaling=True,
    include_top=True,
    classes=3
)
densenet.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train your model
densenet.fit(train_dataset, validation_data=test_dataset)
```

## Contributors
If you'd like to contribute, please see our [contributing guide](.github/CONTRIBUTING.md).

To find an issue to tackle, please check our [call for contributions](.github/CALL_FOR_CONTRIBUTIONS.md).

We would like to leverage/outsource the Keras community not only for bug reporting,
but also for active development for feature delivery. To achieve this, here is the predefined
process for how to contribute to this repository:

1) Contributors are always welcome to help us fix an issue, add tests, better documentation.  
2) If contributors would like to create a backbone, we usually require a pre-trained weight set
with the model for one dataset as the first PR, and a training script as a follow-up. The training script will preferrably help us reproduce the results claimed from paper. The backbone should be generic but the training script can contain paper specific parameters such as learning rate schedules and weight decays. The training script will be used to produce leaderboard results.  
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
Note that in some the 3D Object Detection layers, custom TF ops are used. The
binaries for these ops are not shipped in our PyPi package in order to keep our
wheels pure-Python.

If you'd like to use these custom ops, you can install from source using the
instructions below.

### Installing KerasCV with Custom Ops from Source

Installing custom ops from source requires the [Bazel](https://bazel.build/) build
system (version >= 5.4.0).  Steps to install Bazel can be [found here](https://github.com/keras-team/keras/blob/v2.11.0/.devcontainer/Dockerfile#L21-L23).

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
StableDiffusion, Vision Transfomer

## Citing KerasCV

If KerasCV helps your research, we appreciate your citations.
Here is the BibTeX entry:

```bibtex
@misc{wood2022kerascv,
  title={KerasCV},
  author={Wood, Luke and Tan, Zhenyu and Stenbit, Ian and Zhu, Scott and Chollet, Fran\c{c}ois and others},
  year={2022},
  howpublished={\url{https://github.com/keras-team/keras-cv}},
}
```
