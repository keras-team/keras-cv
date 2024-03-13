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

from keras_cv.api_export import keras_cv_export
from keras_cv.backend import keras
from keras_cv.bounding_box.converters import _decode_deltas_to_boxes
from keras_cv.bounding_box.utils import _clip_boxes
from keras_cv.layers.object_detection.anchor_generator import AnchorGenerator
from keras_cv.layers.object_detection.roi_generator import ROIGenerator
from keras_cv.layers.object_detection.roi_align import _ROIAligner
from keras_cv.models.object_detection.faster_rcnn import FeaturePyramid
from keras_cv.models.object_detection.faster_rcnn import RPNHead
from keras_cv.models.task import Task
from keras_cv.utils.train import get_feature_extractor

BOX_VARIANCE = [0.1, 0.1, 0.2, 0.2]


# TODO(tanzheny): add more configurations
@keras_cv_export("keras_cv.models.FasterRCNN")
class FasterRCNN(Task):
    def __init__(
        self,
        batch_size,
        backbone,
        num_classes,
        bounding_box_format,
        anchor_generator=None,
        feature_pyramid=None,
        *args,
        **kwargs,
    ):

        # Create the Input Layer
        extractor_levels = ["P2", "P3", "P4", "P5"]
        extractor_layer_names = [
            backbone.pyramid_level_inputs[i] for i in extractor_levels
        ]
        feature_extractor = get_feature_extractor(
            backbone, extractor_layer_names, extractor_levels
        )
        feature_pyramid = feature_pyramid or FeaturePyramid()
        image_shape = feature_extractor.input_shape[1:]  # exclude the batch size
        images = keras.layers.Input(
            image_shape, batch_size=batch_size, name="images"
        )
        print(f"{image_shape=}")
        print(f"{images.shape=}")

        # Get the backbone outputs
        backbone_outputs = feature_extractor(images)
        feature_map = feature_pyramid(backbone_outputs)
        print("backbone_outputs")
        for key, value in backbone_outputs.items():
            print(f"\t{key}: {value.shape}")
        print("feature_map")
        for key, value in feature_map.items():
            print(f"\t{key}: {value.shape}")

        # Get the Region Proposal Boxes and Scores
        scales = [2**x for x in [0]]
        aspect_ratios = [0.5, 1.0, 2.0]
        num_anchors_per_location = len(scales) * len(aspect_ratios)
        rpn_head = RPNHead(num_anchors_per_location=num_anchors_per_location)
        # [BS, num_anchors, 4], [BS, num_anchors, 1]
        rpn_boxes, rpn_scores = rpn_head(feature_map)
        print("rpn_boxes")
        for key, value in rpn_boxes.items():
            print(f"\t{key}: {value.shape}")
        print("rpn_scores")
        for key, value in rpn_scores.items():
            print(f"\t{key}: {value.shape}")

        # Create the anchors
        anchor_generator = anchor_generator or AnchorGenerator(
            bounding_box_format=bounding_box_format,
            sizes={
                "P2": 32.0,
                "P3": 64.0,
                "P4": 128.0,
                "P5": 256.0,
                "P6": 512.0,
            },
            scales=scales,
            aspect_ratios=aspect_ratios,
            strides={f"P{i}": 2**i for i in range(2, 7)},
            clip_boxes=True,
        )
        # Note: `image_shape` should not be of NoneType
        # Need to assert before this line
        anchors = anchor_generator(image_shape=image_shape)
        print("anchors")
        for key, value in anchors.items():
            print(f"\t{key}: {value.shape}")

        # decode the deltas to boxes
        decoded_rpn_boxes = _decode_deltas_to_boxes(
            anchors=anchors,
            boxes_delta=rpn_boxes,
            anchor_format=bounding_box_format,
            box_format=bounding_box_format,
            variance=BOX_VARIANCE,
        )
        print("decoded_rpn_boxes")
        for key, value in decoded_rpn_boxes.items():
            print(f"\t{key}: {value.shape}")

        # Generate the Region of Interests
        roi_generator = ROIGenerator(
            bounding_box_format=bounding_box_format,
            nms_score_threshold_train=float("-inf"),
            nms_score_threshold_test=float("-inf"),
        )
        rois, _ = roi_generator(decoded_rpn_boxes, rpn_scores)
        rois = _clip_boxes(rois, bounding_box_format, image_shape)
        print(f"{rois.shape=}")

        # Using the regions call the rcnn head
        roi_pooler = _ROIAligner(bounding_box_format="yxyx")
        feature_map = roi_pooler(features=feature_map, boxes=rois)
        print(f"{feature_map.shape=}")
        
        #
        # # Create the anchor generator
        # scales = [2**x for x in [0]]
        # aspect_ratios = [0.5, 1.0, 2.0]
        # anchor_generator = anchor_generator or AnchorGenerator(
        #     bounding_box_format="yxyx",
        #     sizes={
        #         "P2": 32.0,
        #         "P3": 64.0,
        #         "P4": 128.0,
        #         "P5": 256.0,
        #         "P6": 512.0,
        #     },
        #     scales=scales,
        #     aspect_ratios=aspect_ratios,
        #     strides={f"P{i}": 2**i for i in range(2, 7)},
        #     clip_boxes=True,
        # )

        # # Create the Region Proposal Network Head
        # num_anchors_per_location = len(scales) * len(aspect_ratios)
        # rpn_head = RPNHead(num_anchors_per_location=num_anchors_per_location)

        # # Create the Region of Interest Generator
        # roi_generator = ROIGenerator(
        #     bounding_box_format="yxyx",
        #     nms_score_threshold_train=float("-inf"),
        #     nms_score_threshold_test=float("-inf"),
        # )

        # # Create the Box Matcher
        # box_matcher = BoxMatcher(
        #     thresholds=[0.0, 0.5], match_values=[-2, -1, 1]
        # )

        # # Create the Region of Interest Sampler

        # images = None
        # box_pred = None
        # class_pred = None
        # inputs = {"images": images}
        # outputs = {"box": box_pred, "classification": class_pred}
        # super().__init__(inputs=inputs, outputs=outputs, *args, **kwargs)

    # def train_step(self, *args):
    #     data = args[-1]
    #     args = args[:-1]
    #     x, y = unpack_input(data)
    #     return super().train_step(*args, (x, y))

    # def test_step(self, *args):
    #     data = args[-1]
    #     args = args[:-1]
    #     x, y = unpack_input(data)
    #     return super().test_step(*args, (x, y))
