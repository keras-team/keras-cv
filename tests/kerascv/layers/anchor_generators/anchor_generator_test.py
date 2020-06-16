# Copyright 2020 The Keras CV Authors
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
from kerascv.layers.anchor_generators import anchor_generator


def test_single_scale_absolute_coordinate():
    anchor_gen = anchor_generator.AnchorGenerator(
        image_size=(300, 300),
        scales=[0.2],
        aspect_ratios=[1.0],
        clip_boxes=False,
        normalize_coordinates=False,
    )
    anchor_out = anchor_gen((2, 2))
    expected_out = np.asarray(
        [
            [45, 45, 105, 105],
            [45, 195, 105, 255],
            [195, 45, 255, 105],
            [195, 195, 255, 255],
        ]
    ).astype(np.float32)
    np.testing.assert_allclose(expected_out, anchor_out)


def test_single_scale_non_square_image():
    anchor_gen = anchor_generator.AnchorGenerator(
        image_size=(300, 200),
        scales=[0.2],
        aspect_ratios=[1.0],
        clip_boxes=False,
        normalize_coordinates=False,
    )
    anchor_out = anchor_gen((2, 2))
    expected_out = np.asarray(
        [[30, 30, 70, 70], [30, 130, 70, 170], [130, 30, 170, 70], [130, 130, 170, 170]]
    ).astype(np.float32)
    np.testing.assert_allclose(expected_out, anchor_out)


def test_single_scale_normalized_coordinate():
    anchor_gen = anchor_generator.AnchorGenerator(
        image_size=(300, 300),
        scales=[0.2],
        aspect_ratios=[1.0],
        clip_boxes=False,
        normalize_coordinates=True,
    )
    anchor_out = anchor_gen((2, 2))
    expected_out = np.asarray(
        [
            [0.15, 0.15, 0.35, 0.35],
            [0.15, 0.65, 0.35, 0.85],
            [0.65, 0.15, 0.85, 0.35],
            [0.65, 0.65, 0.85, 0.85],
        ]
    ).astype(np.float32)
    np.testing.assert_allclose(expected_out, anchor_out)


def test_single_scale_customized_stride():
    anchor_gen = anchor_generator.AnchorGenerator(
        image_size=(300, 300),
        scales=[0.2],
        aspect_ratios=[1.0],
        stride=[100, 100],
        clip_boxes=False,
        normalize_coordinates=False,
    )
    anchor_out = anchor_gen((2, 2))
    # The center of absolute anchor points would be [50, 50], [50, 150], [150, 50] and [150, 150]
    expected_out = np.asarray(
        [[20, 20, 80, 80], [20, 120, 80, 180], [120, 20, 180, 80], [120, 120, 180, 180]]
    ).astype(np.float32)
    np.testing.assert_allclose(expected_out, anchor_out)


def test_single_scale_customized_offset():
    anchor_gen = anchor_generator.AnchorGenerator(
        image_size=(300, 300),
        scales=[0.2],
        aspect_ratios=[1.0],
        offset=[0.3, 0.3],
        clip_boxes=False,
        normalize_coordinates=False,
    )
    anchor_out = anchor_gen((2, 2))
    # The first center of absolute anchor points would be 300 / 2 * 0.3 = 45, the second would be 45 + 150 = 195
    expected_out = np.asarray(
        [[15, 15, 75, 75], [15, 165, 75, 225], [165, 15, 225, 75], [165, 165, 225, 225]]
    ).astype(np.float32)
    np.testing.assert_allclose(expected_out, anchor_out)


def test_over_scale_absolute_coordinate_no_clip():
    anchor_gen = anchor_generator.AnchorGenerator(
        image_size=(300, 300),
        scales=[0.7],
        aspect_ratios=[1.0],
        clip_boxes=False,
        normalize_coordinates=False,
    )
    anchor_out = anchor_gen((2, 2))
    expected_out = np.asarray(
        [
            [-30, -30, 180, 180],
            [-30, 120, 180, 330],
            [120, -30, 330, 180],
            [120, 120, 330, 330],
        ]
    ).astype(np.float32)
    np.testing.assert_allclose(expected_out, anchor_out)


def test_over_scale_absolute_coordinate_clip():
    anchor_gen = anchor_generator.AnchorGenerator(
        image_size=(300, 300),
        scales=[0.7],
        aspect_ratios=[1.0],
        clip_boxes=True,
        normalize_coordinates=False,
    )
    anchor_out = anchor_gen((2, 2))
    expected_out = np.asarray(
        [[0, 0, 180, 180], [0, 120, 180, 300], [120, 0, 300, 180], [120, 120, 300, 300]]
    ).astype(np.float32)
    np.testing.assert_allclose(expected_out, anchor_out)


def test_over_scale_normalized_coordinate_no_clip():
    anchor_gen = anchor_generator.AnchorGenerator(
        image_size=(300, 300),
        scales=[0.7],
        aspect_ratios=[1.0],
        clip_boxes=False,
        normalize_coordinates=True,
    )
    anchor_out = anchor_gen((2, 2))
    expected_out = np.asarray(
        [
            [-0.1, -0.1, 0.6, 0.6],
            [-0.1, 0.4, 0.6, 1.1],
            [0.4, -0.1, 1.1, 0.6],
            [0.4, 0.4, 1.1, 1.1],
        ]
    ).astype(np.float32)
    np.testing.assert_allclose(expected_out, anchor_out)


def test_over_scale_normalized_coordinate_clip():
    anchor_gen = anchor_generator.AnchorGenerator(
        image_size=(300, 300),
        scales=[0.7],
        aspect_ratios=[1.0],
        clip_boxes=True,
        normalize_coordinates=True,
    )
    anchor_out = anchor_gen((2, 2))
    expected_out = np.asarray(
        [
            [0.0, 0.0, 0.6, 0.6],
            [0.0, 0.4, 0.6, 1.0],
            [0.4, 0.0, 1.0, 0.6],
            [0.4, 0.4, 1.0, 1.0],
        ]
    ).astype(np.float32)
    np.testing.assert_allclose(expected_out, anchor_out)


def test_multi_aspect_ratios():
    anchor_gen = anchor_generator.AnchorGenerator(
        image_size=(300, 300),
        scales=[0.2, 0.2],
        aspect_ratios=[0.64, 1.0],
        clip_boxes=False,
        normalize_coordinates=False,
    )
    anchor_out = anchor_gen((2, 2))
    # height is 300 * 0.2 / 0.8 = 75; width is 300 * 0.2 * 0.8 = 48
    expected_out = np.asarray(
        [
            [37.5, 51.0, 112.5, 99.0],
            [45.0, 45.0, 105.0, 105.0],
            [37.5, 201.0, 112.5, 249.0],
            [45.0, 195.0, 105.0, 255.0],
            [187.5, 51.0, 262.5, 99.0],
            [195.0, 45.0, 255.0, 105.0],
            [187.5, 201.0, 262.5, 249.0],
            [195.0, 195.0, 255.0, 255.0],
        ]
    ).astype(np.float32)
    np.testing.assert_allclose(expected_out, anchor_out)


def test_multi_scales():
    anchor_gen = anchor_generator.AnchorGenerator(
        image_size=(300, 300),
        scales=[0.2, 0.5],
        aspect_ratios=[1.0, 1.0],
        clip_boxes=False,
        normalize_coordinates=False,
    )
    anchor_out = anchor_gen((2, 2))
    expected_out = np.asarray(
        [
            [45.0, 45.0, 105.0, 105.0],
            [0.0, 0.0, 150.0, 150.0],
            [45.0, 195.0, 105.0, 255.0],
            [0.0, 150.0, 150.0, 300.0],
            [195.0, 45.0, 255.0, 105.0],
            [150.0, 0.0, 300.0, 150.0],
            [195.0, 195.0, 255.0, 255.0],
            [150.0, 150.0, 300.0, 300.0],
        ]
    ).astype(np.float32)
    np.testing.assert_allclose(expected_out, anchor_out)


def test_config_with_custom_name():
    layer = anchor_generator.AnchorGenerator(
        (300, 300), [1.0], [1.0], name="anchor_generator"
    )
    config = layer.get_config()
    layer_1 = anchor_generator.AnchorGenerator.from_config(config)
    np.testing.assert_equal(layer_1.name, layer.name)
