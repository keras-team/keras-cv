import numpy as np
from kerascv.layers.ssd_box_coder import SSDBoxCoder


def test_encode_decode_variance():
    gt_boxes = np.asarray([[10.0, 10.0, 20.0, 15.0], [0.2, 0.1, 0.5, 0.4]], np.float32)
    anchors = np.array([[15.0, 12.0, 30.0, 18.0], [0.1, 0.0, 0.7, 0.9]], np.float32)
    encode_layer = SSDBoxCoder(
        center_variances=[0.5, 1 / 3], size_variances=[0.25, 0.2]
    )
    encoded_gt_boxes = encode_layer(gt_boxes, anchors)
    expected_out = np.asarray(
        [
            [-1.0, -1.25, -1.62186, -0.911608],
            [-0.166667, -0.666667, -2.772588, -5.493062],
        ]
    )
    np.testing.assert_allclose(expected_out, encoded_gt_boxes, rtol=1e-06, atol=1e-6)

    decode_layer = SSDBoxCoder(
        center_variances=[0.5, 1 / 3], size_variances=[0.25, 0.2], invert=True
    )
    decoded_gt_boxes = decode_layer(encoded_gt_boxes, anchors)
    np.testing.assert_allclose(gt_boxes, decoded_gt_boxes, rtol=1e-6, atol=1e-6)


def test_encode_decode_no_variance():
    gt_boxes = np.asarray([[10.0, 10.0, 20.0, 15.0], [0.2, 0.1, 0.5, 0.4]], np.float32)
    anchors = np.array([[15.0, 12.0, 30.0, 18.0], [0.1, 0.0, 0.7, 0.9]], np.float32)
    encode_layer = SSDBoxCoder()
    encoded_gt_boxes = encode_layer(gt_boxes, anchors)
    expected_out = np.asarray(
        [[-0.5, -0.41666, -0.40546, -0.18232], [-0.08333, -0.22222, -0.69314, -1.0986]]
    )
    np.testing.assert_allclose(expected_out, encoded_gt_boxes, rtol=1e-05, atol=1e-5)

    decode_layer = SSDBoxCoder(invert=True)
    decoded_gt_boxes = decode_layer(encoded_gt_boxes, anchors)
    np.testing.assert_allclose(gt_boxes, decoded_gt_boxes, rtol=1e-6, atol=1e-6)


def test_config_with_custom_name():
    layer = SSDBoxCoder(
        center_variances=[0.1, 0.1], size_variances=[0.2, 0.2], name="box_coder"
    )
    config = layer.get_config()
    layer_1 = SSDBoxCoder.from_config(config)
    np.testing.assert_equal(layer_1.name, layer.name)
