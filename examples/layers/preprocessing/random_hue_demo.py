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
"""random_hue_demo.py shows how to use the RandomHue preprocessing layer.
Operates on the oxford_flowers102 dataset.  In this script the flowers
are loaded, then are passed through the preprocessing layers.
Finally, they are shown using matplotlib.
"""
import utils

from keras_cv.layers import preprocessing


def main():
    # Prepare flower dataset dataset.
    train_ds = utils.prepare_dataset()

    # Prepare augmentation layer.
    random_hue = preprocessing.RandomHue(factor=(0.0, 1.0), value_range=(0, 255))

    # Apply augmentation.
    train_ds = train_ds.map(lambda x, y: (random_hue(x), y))

    # visualize.
    utils.visualize_dataset(train_ds)


if __name__ == "__main__":
    main()
