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
from keras_cv.backend import keras
from keras_cv.backend.config import multi_backend

if multi_backend():
    from keras_core.src.backend import vectorized_map  # noqa: F403, F401
    from keras_core.src.ops import *  # noqa: F403, F401
    from keras_core.src.utils.image_utils import (  # noqa: F403, F401
        smart_resize,
    )
else:
    from keras_cv.backend.tf_ops import *  # noqa: F403, F401


# TODO(ianstenbit): Upstream to Keras Core and remove this unholy hack
def segment_max(data, segment_ids, num_segments=None, sorted=False):
    if multi_backend():
        if keras.backend.backend() == "jax":
            return segment_max_jax(data, segment_ids, num_segments, sorted)
        if keras.backend.backend() == "torch":
            raise NotImplementedError("Torch does not support segment_max yet.")

    # Otherwise we can use the TF version
    return segment_max_tf(data, segment_ids, num_segments, sorted)


def segment_max_jax(data, segment_ids, num_segments=None, sorted=False):
    import jax

    if num_segments is None:
        raise ValueError(
            "Argument `num_segments` must be set when using the JAX backend. "
            "Received: num_segments=None"
        )
    return jax.ops.segment_max(
        data, segment_ids, num_segments, indices_are_sorted=sorted
    )


def segment_max_tf(data, segment_ids, num_segments=None, sorted=False):
    import tensorflow as tf

    if sorted:
        return tf.math.segment_max(data, segment_ids)
    else:
        if num_segments is None:
            unique_segment_ids, _ = tf.unique(segment_ids)
            num_segments = tf.shape(unique_segment_ids)[0]
        return tf.math.unsorted_segment_max(data, segment_ids, num_segments)
