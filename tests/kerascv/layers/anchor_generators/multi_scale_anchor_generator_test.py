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
from kerascv.layers.anchor_generators import multi_scale_anchor_generator


def test_single_feature_map_absolute_coordinate():
    anchor_gen = multi_scale_anchor_generator.MultiScaleAnchorGenerator(
        image_size=(300, 300),
        scales=[[0.2]],
        aspect_ratios=[[1.0]],
        clip_boxes=False,
        norm_coord=False,
    )
    anchor_out = anchor_gen([(2, 2)])
    expected_out = np.asarray(
        [
            [45, 45, 105, 105],
            [45, 195, 105, 255],
            [195, 45, 255, 105],
            [195, 195, 255, 255],
        ]
    ).astype(np.float32)
    np.testing.assert_allclose(expected_out, anchor_out)


def test_single_feature_map_multi_aspect_ratios():
    anchor_gen = multi_scale_anchor_generator.MultiScaleAnchorGenerator(
        image_size=(300, 300),
        scales=[[0.2, 0.2]],
        aspect_ratios=[[0.64, 1.0]],
        clip_boxes=False,
        norm_coord=False,
    )
    anchor_out = anchor_gen([(2, 2)])
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


def test_multi_feature_maps_absolute_coordinate():
    anchor_gen = multi_scale_anchor_generator.MultiScaleAnchorGenerator(
        image_size=(300, 300),
        scales=[[0.1], [0.2]],
        aspect_ratios=[[1.0], [1.0]],
        clip_boxes=False,
        norm_coord=False,
    )
    anchor_out = anchor_gen([(3, 3), (2, 2)])
    # The first height and width is 30, the second height and width is 60.
    expected_out = np.asarray(
        [
            [35.0, 35.0, 65.0, 65.0],
            [35.0, 135.0, 65.0, 165.0],
            [35.0, 235.0, 65.0, 265.0],
            [135.0, 35.0, 165.0, 65.0],
            [135.0, 135.0, 165.0, 165.0],
            [135.0, 235.0, 165.0, 265.0],
            [235.0, 35.0, 265.0, 65.0],
            [235.0, 135.0, 265.0, 165.0],
            [235.0, 235.0, 265.0, 265.0],
            [45.0, 45.0, 105.0, 105.0],
            [45.0, 195.0, 105.0, 255.0],
            [195.0, 45.0, 255.0, 105.0],
            [195.0, 195.0, 255.0, 255.0],
        ]
    ).astype(np.float32)
    np.testing.assert_allclose(expected_out, anchor_out)


def test_multi_feature_maps_customized_stride():
    anchor_gen = multi_scale_anchor_generator.MultiScaleAnchorGenerator(
        image_size=(300, 300),
        scales=[[0.1], [0.2]],
        aspect_ratios=[[1.0], [1.0]],
        anchor_strides=[[120, 120], [160, 160]],
        clip_boxes=False,
        norm_coord=False,
    )
    anchor_out = anchor_gen([(3, 3), (2, 2)])
    # The first center of anchor point for the first feature map is 120 * 0.5 = 60, then 180
    # The first center of anchor point for the second feature map is 160 * 0.5 = 80, then 240
    expected_out = np.asarray(
        [
            [45.0, 45.0, 75.0, 75.0],
            [45.0, 165.0, 75.0, 195.0],
            [45.0, 285.0, 75.0, 315.0],
            [165.0, 45.0, 195.0, 75.0],
            [165.0, 165.0, 195.0, 195.0],
            [165.0, 285.0, 195.0, 315.0],
            [285.0, 45.0, 315.0, 75.0],
            [285.0, 165.0, 315.0, 195.0],
            [285.0, 285.0, 315.0, 315.0],
            [50.0, 50.0, 110.0, 110.0],
            [50.0, 210.0, 110.0, 270.0],
            [210.0, 50.0, 270.0, 110.0],
            [210.0, 210.0, 270.0, 270.0],
        ]
    ).astype(np.float32)
    np.testing.assert_allclose(expected_out, anchor_out)


def test_multi_feature_maps_customized_offset():
    anchor_gen = multi_scale_anchor_generator.MultiScaleAnchorGenerator(
        image_size=(300, 300),
        scales=[[0.1], [0.2]],
        aspect_ratios=[[1.0], [1.0]],
        anchor_offsets=[[0.2, 0.2], [0.3, 0.3]],
        clip_boxes=False,
        norm_coord=False,
    )
    anchor_out = anchor_gen([(3, 3), (2, 2)])
    # The first center of anchor point for the first feature map is 100 * 0.2 = 20, then 120
    # The first center of anchor point for the second feature map is 150 * 0.3 = 45, then 195
    expected_out = np.asarray(
        [
            [5.0, 5.0, 35.0, 35.0],
            [5.0, 105.0, 35.0, 135.0],
            [5.0, 205.0, 35.0, 235.0],
            [105.0, 5.0, 135.0, 35.0],
            [105.0, 105.0, 135.0, 135.0],
            [105.0, 205.0, 135.0, 235.0],
            [205.0, 5.0, 235.0, 35.0],
            [205.0, 105.0, 235.0, 135.0],
            [205.0, 205.0, 235.0, 235.0],
            [15, 15, 75, 75],
            [15, 165, 75, 225],
            [165, 15, 225, 75],
            [165, 165, 225, 225],
        ]
    ).astype(np.float32)
    np.testing.assert_allclose(expected_out, anchor_out)


def test_multi_feature_maps_over_scale_absolute_coordinate_no_clip():
    anchor_gen = multi_scale_anchor_generator.MultiScaleAnchorGenerator(
        image_size=(300, 300),
        scales=[[0.1], [0.7]],
        aspect_ratios=[[1.0], [1.0]],
        clip_boxes=False,
        norm_coord=False,
    )
    anchor_out = anchor_gen([(3, 3), (2, 2)])
    # The first height and width is 30, the second height and width is 60.
    expected_out = np.asarray(
        [
            [35.0, 35.0, 65.0, 65.0],
            [35.0, 135.0, 65.0, 165.0],
            [35.0, 235.0, 65.0, 265.0],
            [135.0, 35.0, 165.0, 65.0],
            [135.0, 135.0, 165.0, 165.0],
            [135.0, 235.0, 165.0, 265.0],
            [235.0, 35.0, 265.0, 65.0],
            [235.0, 135.0, 265.0, 165.0],
            [235.0, 235.0, 265.0, 265.0],
            [-30, -30, 180, 180],
            [-30, 120, 180, 330],
            [120, -30, 330, 180],
            [120, 120, 330, 330],
        ]
    ).astype(np.float32)
    np.testing.assert_allclose(expected_out, anchor_out)


def test_multi_feature_maps_over_scale_absolute_coordinate_clip():
    anchor_gen = multi_scale_anchor_generator.MultiScaleAnchorGenerator(
        image_size=(300, 300),
        scales=[[0.1], [0.7]],
        aspect_ratios=[[1.0], [1.0]],
        clip_boxes=True,
        norm_coord=False,
    )
    anchor_out = anchor_gen([(3, 3), (2, 2)])
    # The first height and width is 30, the second height and width is 60.
    expected_out = np.asarray(
        [
            [35.0, 35.0, 65.0, 65.0],
            [35.0, 135.0, 65.0, 165.0],
            [35.0, 235.0, 65.0, 265.0],
            [135.0, 35.0, 165.0, 65.0],
            [135.0, 135.0, 165.0, 165.0],
            [135.0, 235.0, 165.0, 265.0],
            [235.0, 35.0, 265.0, 65.0],
            [235.0, 135.0, 265.0, 165.0],
            [235.0, 235.0, 265.0, 265.0],
            [0, 0, 180, 180],
            [0, 120, 180, 300],
            [120, 0, 300, 180],
            [120, 120, 300, 300],
        ]
    ).astype(np.float32)
    np.testing.assert_allclose(expected_out, anchor_out)


def test_config_with_custom_name():
    layer = multi_scale_anchor_generator.MultiScaleAnchorGenerator(
        (300, 300), [[1.0]], [[1.0]], name="multi_anchor_generator"
    )
    config = layer.get_config()
    layer_1 = multi_scale_anchor_generator.MultiScaleAnchorGenerator.from_config(config)
    np.testing.assert_equal(layer_1.name, layer.name)
