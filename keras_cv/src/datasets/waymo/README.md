### The Waymo Open Dataset in keras_cv

To load the Waymo Open Dataset in KerasCV, you'll need to have TF 2.10-compatible version of the `waymo_open_dataset` package installed.

Waymo does not yet offer a 2.10-compatible version of this package on PyPi. As a temporary solution, we offer pre-built Linux wheels for `waymo_open_dataset` for Python [3.8](https://storage.googleapis.com/keras-cv/waymo-open-dataset/waymo_open_dataset_tf_2_10_0-1.4.9-cp38-cp38-linux_x86_64.whl) and [3.10](https://storage.googleapis.com/keras-cv/waymo-open-dataset/waymo_open_dataset_tf_2_10_0-1.4.9-cp310-cp310-linux_x86_64.whl). To install the package for other systems or versions, you can install from source using the instructions in the Waymo Open Dataset [GitHub repo](https://github.com/waymo-research/waymo-open-dataset) after patching in the change from [PR 562](https://github.com/waymo-research/waymo-open-dataset/pull/562).
