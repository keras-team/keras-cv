[![](https://github.com/keras-team/keras-cv/workflows/Tests/badge.svg?branch=master)](https://github.com/keras-team/keras-cv/actions?query=workflow%3ATests+branch%3Amaster)
![Downloads](https://img.shields.io/pypi/dm/keras-cv.svg)
![Python](https://img.shields.io/badge/python-v3.7.0+-success.svg)
![Tensorflow](https://img.shields.io/badge/tensorflow-v2.9.0+-success.svg)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/keras-team/keras-cv/issues)

# Mission

KerasCV is a repository of modular building blocks (layers, metrics, losses, data-augmentation) that
applied computer vision engineers can leverage to quickly assemble production-grade, state-of-the-art
training and inference pipelines for common use cases such as image classification, object detection,
image segmentation, image data augmentation, etc. It also provides standardaized APIs for some core
computer vision concepts such as bounding boxes, keypoints, point clouds, etc.

KerasCV provides the following values for users:
- modular mid-level APIs and composable meta architectures
- mixed-precision and xla enabled components
- highly optimized models, compatible between GPUs and TPUs
- reproducible training results and leaderboard
- useful tools for evaluation, visualization and explanation.
- source for inference conversion (TFLite, edge devices, TensorRT, etc)

KerasCV can be understood as a horizontal extension of the Keras API: the components are new first-party
Keras objects (layers, metrics, etc) that are too specialized to be added to core Keras, but that receive
the same level of polish and backwards compatibility guarantees as the rest of the Keras API and that
are maintained by the Keras team itself (unlike TFAddons).

KerasCV's primary goal is to provide a coherent, elegant, and pleasant API to train state of the art computer vision models.
Users should be able to train state of the art models using only `Keras`, `KerasCV`, and TensorFlow core (i.e. `tf.data`) components.

Different from Keras IO, this product focus on meta architectures and training scripts to help users reproduce result from open datasets.

To learn more about the future project direction, please check the [roadmap](.github/ROADMAP.md).

## Quick Links
- [Contributing Guide](.github/CONTRIBUTING.md)
- [Call for Contributions](https://github.com/keras-team/keras-cv/issues?q=is%3Aopen+is%3Aissue+label%3Acontribution-welcome)
- [Roadmap](.github/ROADMAP.md)
- [API Design Guidelines](.github/API_DESIGN.md)

## Contributors
If you'd like to contribute, please see our [contributing guide](.github/CONTRIBUTING.md).

To find an issue to tackle, please check our [call for contributions](.github/CALL_FOR_CONTRIBUTIONS.md).

We would like to leverage/outsource the Keras community not only for bug reporting,
but also for active development for feature delivery. To achieve this, here is the predefined
process for how to contribute to this repository:

1) Contributors are always welcome to help us fix an issue, add tests, better documentation.  
2) If contributors would like to create a backbone, we usually require a pre-trained weight
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


## Installing Custom Ops from Source
Installing from source requires the [Bazel](https://bazel.build/) build system
(version >= 1.0.0).

```
git clone https://github.com/keras-team/keras-cv.git
cd keras-cv

python3 build_deps/configure.py

bazel build build_pip_pkg
bazel-bin/build_pip_pkg wheels

pip install wheels/keras-cv-*.whl
```

## Pretrained Weights
Many models in KerasCV come with pre-trained weights. With the exception of StableDiffusion,
all of these weights are trained using Keras and KerasCV components and training scripts in this
repository. Models may not be trained with the same parameters or preprocessing pipeline
described in their original papers. Performance metrics for pre-trained weights can be found
in the training history for each task. For example, see ImageNet classification training
history for backbone models [here](examples/training/classification/imagenet/training_history.json).
All results are reproducible using the training scripts in this repository. Pre-trained weights
operate on images that have been rescaled using a simple `1/255` rescaling layer.

## Disclaimer

KeraCV provides access to pre-trained models via the `keras_cv.models` API.
These pre-trained models are provided on an "as is" basis, without warranties
or conditions of any kind.
The following underlying models are provided by third parties, and subject to separate licenses:
StableDiffusion

## Citing KerasCV

If KerasCV helps your research, we appreciate your citations.
Here is the BibTeX entry:

```bibtex
@misc{wood2022kerascv,
  title={KerasCV},
  author={Wood, Luke and Tan, Zhenyu and Ian, Stenbit and Zhu, Scott and Chollet, Fran\c{c}ois and others},
  year={2022},
  howpublished={\url{https://github.com/keras-team/keras-cv}},
}
```
