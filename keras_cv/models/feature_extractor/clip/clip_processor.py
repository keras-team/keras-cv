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
from keras_nlp.layers import StartEndPacker

from keras_cv.api_export import keras_cv_export
from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.models.feature_extractor.clip.clip_tokenizer import CLIPTokenizer


@keras_cv_export("keras_cv.models.feature_extractors.CLIPProcessor")
class CLIPProcessor:
    """
    CLIPProcessor is a utility class that provides functionality for processing
    images and texts in the context of the CLIP (Contrastive Language-Image
    Pretraining) model.

    Args:
        input_resolution (int): The resolution of input images.
        vocabulary (str): string or dict, maps token to integer ids. If it is a
            string, it should be the file path to a json file.
        merges: string or list, contains the merge rule. If it is a string, it
            should be the file path to merge rules. The merge rule file should
            have one merge rule per line.

    Methods:
        process_images(image_path: List[str]): Transforms an image located at
            the specified path.

        process_texts(texts: Union[str, List[str]], context_length: int = 77):
            Processes a single text or a list of texts, returning packed token
            sequences.

    """

    def __init__(self, input_resolution, vocabulary, merges, **kwargs):
        self.input_resolution = input_resolution
        self.vocabulary = vocabulary
        self.merges = merges
        self.image_transform = self.transform_image
        self.tokenizer = CLIPTokenizer(
            vocabulary=self.vocabulary,
            merges=self.merges,
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
        mean = ops.array([0.48145466, 0.4578275, 0.40821073])
        std = ops.array([0.26862954, 0.26130258, 0.27577711])

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

    def process_texts(self, texts, context_length: int = 77):
        if isinstance(texts, str):
            texts = [texts]

        def pack_tokens(text):
            return self.packer(
                self.tokenizer(text),
                sequence_length=context_length,
                add_start_value=True,
                add_end_value=True,
            )

        return pack_tokens(texts)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_resolution": self.input_resolution,
                "vocabulary": self.vocabulary,
                "merges": self.merges,
            }
        )
        return config
