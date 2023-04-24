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
# limitations under the License

import tensorflow as tf

from keras_cv.bounding_box import iou
from keras_cv.layers.object_detection.nms import nms


def nms_mod(scores_box, box1, box2, iou_0, i):
    box_iou = iou.compute_iou(
        box1, box2, "center_xywh"
    )  # bounding box format used is CENTER_XYWH
    if box_iou >= iou_0:
        scores_box[i][-1] = (1 - box_iou) * scores_box[i][-1]
    return scores_box


class soft_nms(nms):
    def __init__(self, scores_box, iou_0):
        super().__init__(scores_box, iou_0)
        self.scores_box = scores_box
        self.iou_0 = iou_0

    def compute_soft_nms(self):
        li = []
        while len(
            self.scores_box
        ):  # while all rows are not sliced from scores_box tensor
            x, max_box = nms.find_max_box(
                self.scores_box
            )  # extract the max conf score bbox in this iteration and its row
            li.append(
                max_box[-1]
            )  # append the max_box of this iteration to final list
            self.scores_box = nms.scores_filter(
                x, self.scores_box
            )  # remove the row of max box from scores_box

            for i in range(len(self.scores_box)):
                box1 = tf.reshape(
                    self.scores_box[i][:-1], [1, 4]
                )  # all other boxes in scores-box except curr max_box
                box2 = tf.reshape(li[-1][:-1], [1, 4])  # target max_box
                self.scores_box = nms_mod(self.scores_box, box1, box2, i)

        return li
