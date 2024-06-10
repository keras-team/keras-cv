from keras_cv.src import bounding_box
from keras_cv.src import layers as cv_layers
from keras_cv.src import losses
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops

from keras_cv.src.layers.object_detection.roi_generator import ROIGenerator

from keras_cv.src.models.task import Task
from keras_cv.src.utils.train import get_feature_extractor
from keras_cv.src.models.object_detection.faster_rcnn import FeaturePyramid
from keras_cv.src.models.object_detection.faster_rcnn import RPNHead


@keras_cv_export(
    ["keras_cv.models.FasterRCNN", "keras_cv.models.object_detection.FasterRCNN"]
)
class FasterRCNN(Task):
    def __init__(
        self,
        backbone,
        num_classes,
        bounding_box_format,
        anchor_generator=None,
        feature_pyramid=None,
        rcnn_head=None,
        label_encoder=None,
        *args,
        **kwargs,
    ):
        
        # 1. Backbone
        extractor_levels = ["P3", "P4", "P5"]
        extractor_layer_names = [
            backbone.pyramid_level_inputs[i] for i in extractor_levels
        ]
        feature_extractor = get_feature_extractor(
            backbone, extractor_layer_names, extractor_levels
        )
        
        # 2. Feature Pyramid
        feature_pyramid = feature_pyramid or FeaturePyramid(
            name="feature_pyramid"
        )
        
        # 3. Anchors
        scales = [2**x for x in [0]]
        aspect_ratios = [0.5, 1.0, 2.0]
        anchor_generator = (
            anchor_generator
            or FasterRCNN.default_anchor_generator(
                scales,
                aspect_ratios, 
                bounding_box_format)
        )
        
        # 4. RPN Head
        num_anchors_per_location = len(scales) * len(aspect_ratios)
        rpn_head = RPNHead(
            num_anchors_per_location
        )
        
        # Begin construction of forward pass
        images = keras.layers.Input(
            feature_extractor.input_shape[1:], name="images"
        )
        
        backbone_outputs = feature_extractor(images)
        feature_map = feature_pyramid(backbone_outputs)
        
        # [BS, num_anchors, 4], [BS, num_anchors, 1]
        rpn_boxes, rpn_scores = rpn_head(feature_map)
        
        
        inputs = {"images": images}
        outputs = {
            'rpn_box': rpn_boxes,
            'rpn_classification': rpn_scores
        }
        
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )
        
    
    @staticmethod
    def default_anchor_generator(scales, aspect_ratios, bounding_box_format):
        strides = [2**i for i in range(3, 8)]
        sizes = [32.0, 64.0, 128.0, 256.0, 512.0]
        return cv_layers.AnchorGenerator(
            bounding_box_format=bounding_box_format,
            sizes=sizes,
            aspect_ratios=aspect_ratios,
            scales=scales,
            strides=strides,
            clip_boxes=True,
            name="anchor_generator",
        )
        
        
        
        