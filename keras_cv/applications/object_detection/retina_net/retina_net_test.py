import tensorflow as tf
from absl.testing import parameterized

import keras_cv


class RetinaNetTest(tf.test.TestCase, parameterized.TestCase):
    def test_retina_net_construction(self):
        _ = keras_cv.applications.RetinaNet(
            num_classes=20,
            bounding_box_format="xywh",
            backbone="resnet50",
            backbone_weights=None,
            include_rescaling=True,
        )

    def test_retina_net_include_rescaling_required_with_default_backbone(self):
        with self.assertRaises(ValueError):
            _ = keras_cv.applications.RetinaNet(
                num_classes=20,
                bounding_box_format="xywh",
                backbone="resnet50",
                backbone_weights=None,
                # Note no include_rescaling is provided
            )

    def test_all_metric_formats_must_match(self):
        retina_net = keras_cv.applications.RetinaNet(
            num_classes=20,
            bounding_box_format="xywh",
            backbone="resnet50",
            backbone_weights=None,
            include_rescaling=True,
        )

        # all metric formats must match
        with self.assertRaises(ValueError):
            retina_net.compile(
                optimizer="adam",
                metrics=[
                    keras_cv.metrics.COCOMeanAveragePrecision(
                        class_ids=range(20),
                        bounding_box_format="xyxy",
                        name="Standard MaP",
                    ),
                    keras_cv.metrics.COCOMeanAveragePrecision(
                        class_ids=range(20),
                        bounding_box_format="rel_xyxy",
                        name="Standard MaP",
                    ),
                ],
            )

    def test_overfits_single_bounding_box(self):
        # TODO(lukewood): overfit a single stacked bounding box, make sure the format
        # is correct
        pass
