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
"""keypoints_demo.py shows how to use geometric preprocessing layer to
augment an image and associated keypoints.

Operates on the AFLWL20003D dataset.  In this script the face images
and their 2D keypoint data are loaded, then are passed through the
preprocessing layers applying random shear, translation or rotation to
them. Finally, they are shown using matplotlib. Augmented keypoints
falling outside the images boundaries are discarded.
"""

import demo_utils
import tensorflow as tf

from keras_cv.layers import preprocessing


def create_custom_pipeline():
    # NOTE: we need to use constant fills not to create additional
    # unmarked keypoints in additional reflections or tiling of the image.
    layers = [
        preprocessing.RandomRotation(
            factor=(-1.0, 1.0),
            keypoint_format="xy",
            fill_mode="constant",
        ),
        preprocessing.RandomTranslation(
            height_factor=0.5,
            width_factor=0,
            keypoint_format="xy",
            fill_mode="constant",
        ),
        preprocessing.RandomTranslation(
            width_factor=0.5,
            height_factor=0,
            keypoint_format="xy",
            fill_mode="constant",
        ),
        preprocessing.RandomShear(
            x_factor=0.5,
            keypoint_format="xy",
            fill_mode="constant",
        ),
        preprocessing.RandomShear(
            y_factor=0.5,
            keypoint_format="xy",
            fill_mode="constant",
        ),
    ]
    return preprocessing.RandomAugmentationPipeline(
        layers=layers, augmentations_per_image=3
    )


def main():
    # since we are using warping transformation, keypoints may go
    # outside of the image, and therefore the input keypoint list
    # should be ragged as their number may vaary after augmentation.
    ds = demo_utils.load_AFLW2000_dataset(batch_size=9, ragged_keypoints=True)
    custom_pipeline = create_custom_pipeline()
    ds = ds.map(custom_pipeline, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_dataset(ds)


if __name__ == "__main__":
    main()
