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

from tensorflow import keras
class ObjectDetectionBaseModel(keras.Model):

    """ObjectDetectionBaseModel performs asynchonous label encoding.

    ObjectDetectionBaseModel invokes the provided `label_encoder` in the `tf.data`
    pipeline to ensure optimal training performance.  This is done by overriding the
    methods `train_on_batch()`, `fit()`, `test_on_batch()`, and `evaluate()`.

    """

    def __init__(self, label_encoder, **kwargs):
        pass
