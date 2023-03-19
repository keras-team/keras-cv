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

from keras_cv.metrics.coco.mean_average_precision import (
    _COCOMeanAveragePrecision,
)

try:
    from keras_cv.metrics.coco.pycoco_wrapper import PyCOCOWrapper
    from keras_cv.metrics.coco.pycoco_wrapper import compute_pycoco_metrics
except ImportError:
    print(
        "You do not have pycocotools installed, so KerasCV pycoco metrics are"
        "not available. Please run `pip install pycocotools`."
    )
    pass
