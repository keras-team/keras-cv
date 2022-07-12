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
"""random_shear_bboxdemo.py shows how to use the RandomShear preprocessing layer
   for object detection.
Operates on the voc dataset.  In this script the images and bboxes
are loaded, then are passed through the preprocessing layers.
Finally, they are shown using matplotlib.
"""
import matplotlib.pyplot as plt
from demo_utils import load_voc_dataset
from demo_utils import visualize_data

from keras_cv.layers import preprocessing

IMG_SIZE = (256, 256)
BATCH_SIZE = 9


def main():
    dataset = load_voc_dataset(
        name="voc/2007",
        batch_size=BATCH_SIZE,
        image_size=(256, 256),
    )
    randomshear = preprocessing.RandomShear(
        x_factor=(0.1, 0.3),
        y_factor=(0.1, 0.3),
        bounding_box_format="rel_xyxy",
    )
    plt.figure(figsize=(20, 20))
    dataset = dataset.map(lambda x: randomshear(x))
    visualize_data(data=dataset, bounding_box_format="rel_xyxy")


if __name__ == "__main__":
    main()
