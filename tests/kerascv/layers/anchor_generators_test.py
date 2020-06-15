import numpy as np
from kerascv.layers.anchor_generators import AnchorGenerator


def test_single_scale_absolute_coordinate():
    anchor_gen = AnchorGenerator(
        image_size=(300, 300), scales=[.2], aspect_ratios=[1.], clip_boxes=False, norm_coord=False)
    anchor_out = anchor_gen((2, 2))
    expected_out = np.asarray(
        [[45, 45, 105, 105],
         [45, 195, 105, 255],
         [195, 45, 255, 105],
         [195, 195, 255, 255]]).astype(np.float32)
    np.testing.assert_allclose(anchor_out, expected_out)


def test_single_scale_normalized_coordinate():
    anchor_gen = AnchorGenerator(
        image_size=(300, 300), scales=[.2], aspect_ratios=[1.], clip_boxes=False, norm_coord=True
    )
    anchor_out = anchor_gen((2, 2))
    expected_out = np.asarray(
        [[.15, .15, .35, .35],
         [.15, .65, .35, .85],
         [.65, .15, .85, .35],
         [.65, .65, .85, .85]]).astype(np.float32)
    np.testing.assert_allclose(anchor_out, expected_out)


def test_over_scale_absolute_coordinate_no_clip():
    anchor_gen = AnchorGenerator(
        image_size=(300, 300), scales=[.7], aspect_ratios=[1.], clip_boxes=False, norm_coord=False
    )
    anchor_out = anchor_gen((2, 2))
    expected_out = np.asarray(
        [[-30, -30, 180, 180],
         [-30, 120, 180, 330],
         [120, -30, 330, 180],
         [120, 120, 330, 330]]).astype(np.float32)
    np.testing.assert_allclose(anchor_out, expected_out)


def test_over_scale_absolute_coordinate_clip():
    anchor_gen = AnchorGenerator(
        image_size=(300, 300), scales=[.7], aspect_ratios=[1.], clip_boxes=True, norm_coord=False
    )
    anchor_out = anchor_gen((2, 2))
    expected_out = np.asarray(
        [[0, 0, 180, 180],
         [0, 120, 180, 300],
         [120, 0, 300, 180],
         [120, 120, 300, 300]]).astype(np.float32)
    np.testing.assert_allclose(anchor_out, expected_out)


def test_over_scale_normalized_coordinate_no_clip():
    anchor_gen = AnchorGenerator(
        image_size=(300, 300), scales=[.7], aspect_ratios=[1.], clip_boxes=False, norm_coord=True
    )
    anchor_out = anchor_gen((2, 2))
    expected_out = np.asarray(
        [[-0.1, -0.1, 0.6, 0.6],
         [-0.1, 0.4, 0.6, 1.1],
         [0.4, -0.1, 1.1, 0.6],
         [0.4, 0.4, 1.1, 1.1]]).astype(np.float32)
    np.testing.assert_allclose(anchor_out, expected_out)


def test_over_scale_normalized_coordinate_clip():
    anchor_gen = AnchorGenerator(
        image_size=(300, 300), scales=[.7], aspect_ratios=[1.], clip_boxes=True, norm_coord=True
    )
    anchor_out = anchor_gen((2, 2))
    expected_out = np.asarray(
        [[0., 0., 0.6, 0.6],
         [0., 0.4, 0.6, 1.0],
         [0.4, 0., 1.0, 0.6],
         [0.4, 0.4, 1.0, 1.0]]).astype(np.float32)
    np.testing.assert_allclose(anchor_out, expected_out)


def test_multi_aspect_ratios():
    anchor_gen = AnchorGenerator(
        image_size=(300, 300), scales=[.2, .2], aspect_ratios=[.64, 1.], clip_boxes=False, norm_coord=False
    )
    anchor_out = anchor_gen((2, 2))
    # height is 300 * 0.2 / 0.8 = 75; width is 300 * 0.2 * 0.8 = 48
    expected_out = np.asarray(
        [[37.5, 51., 112.5,  99.],
         [45., 45., 105., 105.],
         [37.5, 201., 112.5, 249.],
         [45., 195., 105., 255.],
         [187.5, 51., 262.5, 99.],
         [195., 45., 255., 105.],
         [187.5, 201., 262.5, 249.],
         [195., 195., 255., 255.]]).astype(np.float32)
    np.testing.assert_allclose(anchor_out, expected_out)


def test_multi_scales():
    anchor_gen = AnchorGenerator(
        image_size=(300, 300), scales=[.2, .5], aspect_ratios=[1., 1.], clip_boxes=False, norm_coord=False
    )
    anchor_out = anchor_gen((2, 2))
    expected_out = np.asarray(
        [[45., 45., 105., 105.],
         [0., 0., 150., 150.],
         [45., 195., 105., 255.],
         [0., 150., 150., 300.],
         [195., 45., 255., 105.],
         [150., 0., 300., 150.],
         [195., 195., 255., 255.],
         [150., 150., 300., 300.]]).astype(np.float32)
    np.testing.assert_allclose(anchor_out, expected_out)


def test_config_with_custom_name():
    layer = AnchorGenerator((300, 300), [1.], [1.], name='anchor_generator')
    config = layer.get_config()
    layer_1 = AnchorGenerator.from_config(config)
    np.testing.assert_equal(layer_1.name, layer.name)
