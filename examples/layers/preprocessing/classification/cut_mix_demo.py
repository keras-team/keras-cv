# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""cut_mix_demo.py shows how to use the CutMix preprocessing layer.

Operates on the oxford_flowers102 dataset.  In this script the flowers
are loaded, then are passed through the preprocessing layers.
Finally, they are shown using matplotlib.
"""

import tensorflow as tf

import examples.layers.preprocessing.classification.demo_utils as demo_utils
from keras_cv import layers


def main():
    cutmix = layers.CutMix()
    ds = demo_utils.load_oxford_dataset()
    ds = ds.map(cutmix, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_dataset(ds)


if __name__ == "__main__":
    main()
