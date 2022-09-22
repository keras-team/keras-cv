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

"""random_resized_crop_demo.py.py shows how to use the RandomResizedCrop
preprocessing layer. Operates on an image of elephant. In this script the image
is loaded, then are passed through the preprocessing layers.
Finally, they are shown using matplotlib.
"""

import demo_utils

from keras_cv.layers import RandomCropAndResize


def main():
    many_elephants = demo_utils.load_elephant_tensor(output_size=(300, 300))
    layer = RandomCropAndResize(
        target_size=(224, 224),
        crop_area_factor=(0.8, 1.0),
        aspect_ratio_factor=(3.0 / 4.0, 4.0 / 3.0),
    )
    augmented = layer(many_elephants)
    demo_utils.gallery_show(augmented.numpy())

    layer = RandomCropAndResize(
        target_size=(224, 224),
        crop_area_factor=(0.01, 1.0),
        aspect_ratio_factor=(3.0 / 4.0, 4.0 / 3.0),
    )
    augmented = layer(many_elephants)
    demo_utils.gallery_show(augmented.numpy())


if __name__ == "__main__":
    main()
