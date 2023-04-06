import numpy as np
import tensorflow as tf

"""
I can't get keras_cv.layers.AnchorGenerator to match these numerics, so I've
copy-pastad this here for now. It needs to go eventually.
"""


def get_feature_sizes(input_shape, pyramid_levels=[3, 7]):
    # https://github.com/google/automl/tree/master/efficientdet/utils.py#L509
    feature_sizes = [input_shape[:2]]
    for _ in range(max(pyramid_levels)):
        pre_feat_size = feature_sizes[-1]
        feature_sizes.append(
            ((pre_feat_size[0] - 1) // 2 + 1, (pre_feat_size[1] - 1) // 2 + 1)
        )  # ceil mode, like padding="SAME" downsampling
    return feature_sizes


def get_anchors(
    input_shape=(512, 512, 3),
    pyramid_levels=[3, 5],
    aspect_ratios=[1],
    num_scales=1,
    anchor_scale=1,
    grid_zero_start=False,
):
    # base anchors
    scales = [2 ** (ii / num_scales) * anchor_scale for ii in range(num_scales)]
    aspect_ratios_tensor = np.array(aspect_ratios, dtype="float32")
    if len(aspect_ratios_tensor.shape) == 1:
        # aspect_ratios = [0.5, 1, 2]
        sqrt_ratios = np.sqrt(aspect_ratios_tensor)
        ww_ratios, hh_ratios = sqrt_ratios, 1 / sqrt_ratios
    else:
        # aspect_ratios = [(1, 1), (1.4, 0.7), (0.7, 1.4)]
        ww_ratios, hh_ratios = (
            aspect_ratios_tensor[:, 0],
            aspect_ratios_tensor[:, 1],
        )
    base_anchors_hh = np.reshape(
        np.expand_dims(scales, 1) * np.expand_dims(hh_ratios, 0), [-1]
    )
    base_anchors_ww = np.reshape(
        np.expand_dims(scales, 1) * np.expand_dims(ww_ratios, 0), [-1]
    )
    base_anchors_hh_half, base_anchors_ww_half = (
        base_anchors_hh / 2,
        base_anchors_ww / 2,
    )
    base_anchors = np.stack(
        [
            base_anchors_hh_half * -1,
            base_anchors_ww_half * -1,
            base_anchors_hh_half,
            base_anchors_ww_half,
        ],
        axis=1,
    )
    # base_anchors = tf.gather(base_anchors, [3, 6, 0, 4, 7, 1, 5, 8, 2])  # re-order according to official generated anchors

    # make grid
    pyramid_levels = list(range(min(pyramid_levels), max(pyramid_levels) + 1))
    feature_sizes = get_feature_sizes(input_shape, pyramid_levels)

    all_anchors = []
    for level in pyramid_levels:
        stride_hh, stride_ww = (
            feature_sizes[0][0] / feature_sizes[level][0],
            feature_sizes[0][1] / feature_sizes[level][1],
        )
        top, left = (
            (0, 0) if grid_zero_start else (stride_hh / 2, stride_ww / 2)
        )
        hh_centers = np.arange(top, input_shape[0], stride_hh)
        ww_centers = np.arange(left, input_shape[1], stride_ww)
        ww_grid, hh_grid = np.meshgrid(ww_centers, hh_centers)
        grid = np.reshape(
            np.stack([hh_grid, ww_grid, hh_grid, ww_grid], 2), [-1, 1, 4]
        )
        anchors = np.expand_dims(
            base_anchors * [stride_hh, stride_ww, stride_hh, stride_ww], 0
        ) + grid.astype(base_anchors.dtype)
        anchors = np.reshape(anchors, [-1, 4])
        all_anchors.append(anchors)
    all_anchors = np.concatenate(all_anchors, axis=0) / [
        input_shape[0],
        input_shape[1],
        input_shape[0],
        input_shape[1],
    ]

    return tf.convert_to_tensor(all_anchors.astype("float32"))
