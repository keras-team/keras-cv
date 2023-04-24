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

import tensorflow as tf

from keras_cv.bounding_box import iou


def find_max_box(scores_box):
    max_score = tf.reduce_max(scores_box[:, -1])
    for i in range(len(scores_box)):
        if scores_box[i, -1] == max_score:
            x = tf.where(scores_box[:, -1])[i].numpy()[0]
            max_box = tf.gather(scores_box, indices=[i])
    return x, max_box


def scores_filter(x, scores_box):
    if x == 0:
        scores_box = scores_box[1:, :]
    elif x == len(scores_box) - 1:
        scores_box = scores_box[:-1, :]
    else:
        scores_box = scores_box[0:x, :] + scores_box[x + 1 :, :]
    return scores_box


def nms_elim(scores_box, box1, box2, iou_0, i):
    box_iou = iou.compute_iou(
        box1, box2, "center_xywh"
    )  # bounding box format used is CENTER_XYWH
    if box_iou > iou_0:
        scores_box = scores_filter(i, scores_box)
    else:
        i = i + 1
    return i, scores_box


class nms:
    def __init__(self, scores_box, iou_0):
        self.scores_box = scores_box
        # tensor of shape [num_bboxes,5] ,
        # 5 : (center_xywh bounding box format points,conf_score)
        self.iou_0 = iou_0  # iou threshold

    def compute_nms(self):
        li = []  # initialise a final empty list

        while len(
            self.scores_box
        ):  # while all rows are not sliced from scores_box tensor
            x, max_box = find_max_box(
                self.scores_box
            )  # extract the max conf score bbox in this iteration and its row
            li.append(
                max_box[-1]
            )  # append the max_box of this iteration to final list
            self.scores_box = scores_filter(
                x, self.scores_box
            )  # remove the row of max box from scores_box

            j = len(self.scores_box)
            ctr = 0

            i = 0

            while ctr != j:
                box1 = tf.reshape(
                    self.scores_box[i][:-1], [1, 4]
                )  # all other boxes in scores-box except curr max_box
                box2 = tf.reshape(li[-1][:-1], [1, 4])  # target max_box
                i_, self.scores_box = nms_elim(
                    self.scores_box, box1, box2, self.iou_0, i
                )
                i = i_
                ctr = ctr + 1
        return li
