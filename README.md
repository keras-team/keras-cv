# KerasCV
[![](https://github.com/keras-team/keras-cv/workflows/Tests/badge.svg?branch=master)](https://github.com/keras-team/keras-cv/actions?query=workflow%3ATests+branch%3Amaster)
![Downloads](https://img.shields.io/pypi/dm/keras-cv.svg)
![Python](https://img.shields.io/badge/python-v3.7.0+-success.svg)
![Tensorflow](https://img.shields.io/badge/tensorflow-v2.8.0+-success.svg)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/keras-team/keras-cv/issues)

KerasCV is a repository of modular building blocks (layers, metrics, losses, data-augmentation) that
applied computer vision engineers can leverage to quickly assemble production-grade, state-of-the-art
training and inference pipelines for common use cases such as image classification, object detection,
image segmentation, image data augmentation, etc.

KerasCV can be understood as a horizontal extension of the Keras API: the components are new first-party
Keras objects (layers, metrics, etc) that are too specialized to be added to core Keras, but that receive
the same level of polish and backwards compatibility guarantees as the rest of the Keras API and that
are maintained by the Keras team itself (unlike TFAddons).

KerasCV's primary goal is to provide a coherent, elegant, and pleasant API to train state of the art computer vision models.
Users should be able to train state of the art models using only `Keras`, `KerasCV`, and TensorFlow core (i.e. `tf.data`) components.

To learn more about the future project direction, please check the [roadmap](.github/ROADMAP.md).

## Quick Links
- [Contributing Guide](.github/CONTRIBUTING.md)
- [Call for Contributions](https://github.com/keras-team/keras-cv/issues?q=is%3Aopen+is%3Aissue+label%3Acontribution-welcome)
- [Roadmap](.github/ROADMAP.md)
- [API Design Guidelines](.github/API_DESIGN.md)

## Contributors
If you'd like to contribute, please see our [contributing guide](.github/CONTRIBUTING.md).

To find an issue to tackle, please check our [call for contributions](.github/CALL_FOR_CONTRIBUTIONS.md).

Thank you to all of our wonderful contributors!

<a href="https://github.com/keras-team/keras-cv/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=keras-team/keras-cv" />
</a>

## Citing KerasCV

If KerasCV helps your research, we appreciate your citations.
Here is the BibTeX entry:

```bibtex
@misc{wood2022kerascv,
  title={KerasCV},
  author={Wood, Luke and Zhu, Scott and Chollet, Fran\c{c}ois and others},
  year={2022},
  howpublished={\url{https://github.com/keras-team/keras-cv}},
}
```
