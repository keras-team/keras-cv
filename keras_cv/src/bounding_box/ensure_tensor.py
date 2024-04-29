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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.utils import preprocessing


@keras_cv_export("keras_cv.bounding_box.ensure_tensor")
def ensure_tensor(boxes, dtype=None):
    boxes = boxes.copy()
    for key in ["boxes", "classes", "confidence"]:
        if key in boxes:
            boxes[key] = preprocessing.ensure_tensor(
                boxes[key],
                dtype=dtype,
            )
    return boxes
