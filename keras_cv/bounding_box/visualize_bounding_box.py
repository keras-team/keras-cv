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

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from keras_cv import bounding_box


def visualize_bounding_boxes_on_image(
    image, bboxes, bounding_box_format=None, text=None, color="r"
):
    if bounding_box_format is None:
        raise ValueError("please specifiy bbox formar")
    else:
        bboxes = bounding_box.convert_format(source=bounding_box_format, target="xyxy")
    fig, ax = plt.subplots()
    ax.imshow(image)
    for box in bboxes:
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=1,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
    plt.show
