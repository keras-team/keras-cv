### The ImageNet Dataset in keras_cv

In order to load ImageNet with KerasCV, you'll need to download the [original ImageNet dataset](https://image-net.org) and parse the images into TFRecords.

Tensorflow provides a [script](https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py) to perform this parsing and upload images to Google Cloud Storage (or optionally to local storage).

Please reference that script's instructions on producing ImageNet TFRecords, and then use the KerasCV loader to load records from wherever you choose to store them.
