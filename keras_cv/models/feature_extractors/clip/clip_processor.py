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
import numpy as np
from keras_nlp.layers import StartEndPacker

from keras_cv.api_export import keras_cv_export
from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.models.feature_extractors.clip.clip_tokenizer import CLIPTokenizer


@keras_cv_export("keras_cv.models.feature_extractors.CLIPProcessor")
class CLIPProcessor:
    def __init__(self, input_resolution):
        self.input_resolution = input_resolution
        self.image_transform = self.transform_image
        self.tokenizer = CLIPTokenizer(
            vocabulary="keras_cv/models/feature_extractors/clip/vocab.json",
            merges="keras_cv/models/feature_extractors/clip/merges.txt",
            unsplittable_tokens=["</w>"],
        )
        self.packer = StartEndPacker(
            start_value=self.tokenizer.token_to_id("<|startoftext|>"),
            end_value=self.tokenizer.token_to_id("<|endoftext|>"),
            pad_value=None,
            sequence_length=77,
            return_padding_mask=True,
        )

    def transform_image(self, image_path):
        input_resolution = self.input_resolution
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])

        image = keras.utils.load_img(image_path)
        image = keras.utils.img_to_array(image)
        image = (
            ops.image.resize(
                image,
                (input_resolution, input_resolution),
                interpolation="bicubic",
            )
            / 255.0
        )
        central_fraction = input_resolution / image.shape[0]
        width, height = image.shape[0], image.shape[1]
        left = ops.cast((width - width * central_fraction) / 2, dtype="int32")
        top = ops.cast((height - height * central_fraction) / 2, dtype="int32")
        right = ops.cast((width + width * central_fraction) / 2, dtype="int32")
        bottom = ops.cast(
            (height + height * central_fraction) / 2, dtype="int32"
        )

        image = ops.slice(
            image, [left, top, 0], [right - left, bottom - top, 3]
        )

        image = (image - mean) / std
        return image

    def process_images(self, images):
        if isinstance(images, str):
            images = [images]

        def process_image(image):
            if isinstance(image, str):
                return self.image_transform(image)

        processed_images = list(map(process_image, images))
        processed_images = ops.stack(processed_images)
        return processed_images

    def process_texts(
        self, texts, context_length: int = 77, truncate: bool = False
    ):
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.token_to_id("<|startoftext|>")
        eot_token = self.tokenizer.token_to_id("<|endoftext|>")
        sot_token = ops.expand_dims(sot_token, axis=-1)
        eot_token = ops.expand_dims(eot_token, axis=-1)

        def pack_tokens(text):
            tok, _ = self.packer(
                self.tokenizer(text),
                sequence_length=context_length,
                add_start_value=sot_token,
                add_end_value=eot_token,
            )
            return ops.concatenate([sot_token, tok, eot_token])

        all_tokens = list(map(pack_tokens, texts))

        result = np.zeros(shape=[len(all_tokens), context_length])

        def process_tokens(i_tokens):
            i, tokens = i_tokens
            if len(tokens) > context_length:
                tokens = tokens[:context_length]

            result[i, : len(tokens)] = tokens

        # Using map with function and passing a list of tuples
        list(map(process_tokens, enumerate(all_tokens)))

        result = ops.stack(result)
        return result
