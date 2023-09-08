# Copyright 2023 The KerasCV Authors
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
"""random_cutout_demo.py shows how to use the RandomCutout preprocessing layer.

Uses the oxford iiit pet_dataset.  In this script the pets
are loaded, then are passed through the preprocessing layers.
Finally, they are shown using matplotlib.
"""
import demo_utils
import tensorflow as tf

from keras_cv.layers import preprocessing


def main():
    ds = demo_utils.load_oxford_iiit_pet_dataset()
    randomcutout = preprocessing.RandomCutout(0.5, 0.5)
    ds = ds.map(randomcutout, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_dataset(ds)


if __name__ == "__main__":
    main()
