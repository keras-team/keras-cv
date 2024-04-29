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
import tree

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import config
from keras_cv.src.backend import keras
from keras_cv.src.models.feature_extractor.clip.clip_processor_utils import (
    convert_inputs_to_list_of_tensor_segments,
)
from keras_cv.src.models.feature_extractor.clip.clip_processor_utils import (
    convert_to_backend_tensor_or_python_list,
)
from keras_cv.src.models.feature_extractor.clip.clip_tokenizer import (
    CLIPTokenizer,
)

try:
    import keras_nlp
    from keras_nlp.layers import StartEndPacker
except ImportError:
    keras_nlp = None
    StartEndPacker = None


@keras_cv_export("keras_cv.models.feature_extractor.CLIPProcessor")
class CLIPProcessor(keras.layers.Layer):
    """
    CLIPProcessor is a utility class that provides functionality for processing
    texts in the context of the CLIP (Contrastive Language-Image
    Pretraining) model.

    Args:
        input_resolution (int): The resolution of input images.
        vocabulary (str): string or dict, maps token to integer ids. If it is a
            string, it should be the file path to a json file.
        merges: string or list, contains the merge rule. If it is a string, it
            should be the file path to merge rules. The merge rule file should
            have one merge rule per line.

    """

    def __init__(self, vocabulary, merges, **kwargs):
        super().__init__(**kwargs)
        if keras_nlp is None:
            raise ValueError(
                "ClipTokenizer requires keras-nlp. Please install "
                "using pip `pip install -U keras-nlp && pip install -U keras`"
            )
        self.vocabulary = vocabulary
        self.merges = merges
        self.tokenizer = CLIPTokenizer(
            vocabulary=self.vocabulary,
            merges=self.merges,
        )
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True

    def build(self, input_shape):
        # Defer packer creation to `build()` so that we can be sure tokenizer
        # assets have loaded when restoring a saved model.
        self.packer = StartEndPacker(
            start_value=self.tokenizer.token_to_id("<|startoftext|>"),
            end_value=self.tokenizer.token_to_id("<|endoftext|>"),
            pad_value=None,
            sequence_length=77,
            return_padding_mask=True,
        )
        self.built = True

    def _process_texts(self, texts, context_length: int = 77):
        # Ensure the layer is built
        if not self.built:
            self.build(None)

        texts = convert_inputs_to_list_of_tensor_segments(texts)

        if len(texts) != 1:
            raise ValueError(
                "CLIP requires each input feature to contain only "
                f"one segment, but received {len(texts)}."
            )

        token_ids, padding_mask = self.packer(
            self.tokenizer(texts[0]),
            sequence_length=context_length,
            add_start_value=True,
            add_end_value=True,
        )
        return {"token_ids": token_ids, "padding_mask": padding_mask}

    def call(self, texts, context_length: int = 77):
        return self._process_texts(texts, context_length=context_length)

    def get_build_config(self):
        return None

    def __call__(self, *args, **kwargs):
        # Always place on CPU for preprocessing, to avoid expensive back and
        # forth copies to GPU before the trainable model.
        with tf.device("cpu"):
            outputs = super().__call__(*args, **kwargs)

            # Jax and Torch lack native string and ragged types.
            # If we are running on those backends and not running with tf.data
            # (we are outside a tf.function), we covert all ragged and string
            # tensor to pythonic types.
            is_tf_backend = config.backend() == "tensorflow"
            is_in_tf_graph = not tf.executing_eagerly()
            if not is_tf_backend and not is_in_tf_graph:
                outputs = tree.map_structure(
                    convert_to_backend_tensor_or_python_list, outputs
                )

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary": self.vocabulary,
                "merges": self.merges,
            }
        )
        return config
