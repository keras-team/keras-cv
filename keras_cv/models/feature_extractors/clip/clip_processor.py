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

from keras_cv.api_export import keras_cv_export
from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.models.feature_extractors.clip.clip_tokenizer import CLIPTokenizer


@keras_cv_export("keras_cv.models.feature_extractors.CLIPProcessor")
class CLIPProcessor:
    def __init__(self, input_resolution):
        self.input_resolution = input_resolution
        self.image_transform = self.transform_image
        self.tokenizer = CLIPTokenizer()

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

        processed_images = []
        for image in images:
            if isinstance(image, str):
                image = self.image_transform(image)
                processed_images.append(image)
        processed_images = ops.stack(processed_images)
        return processed_images

    def process_texts(
        self, texts, context_length: int = 77, truncate: bool = False
    ):
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        all_tokens = [
            [sot_token] + self.tokenizer.encode(text) + [eot_token]
            for text in texts
        ]

        result = np.zeros(shape=[len(all_tokens), context_length])

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(
                        f"Input {texts[i]} is too long for context "
                        "length {context_length}"
                    )
            result[i, : len(tokens)] = tokens

        result = ops.stack(result)
        return result

        images = self.process_images(images)
        texts = self.process_texts(texts)
        return (images, texts)
