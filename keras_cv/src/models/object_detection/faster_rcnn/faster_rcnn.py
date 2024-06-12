import tree

from keras_cv.src import bounding_box
from keras_cv.src import layers as cv_layers
from keras_cv.src import losses
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops

from keras_cv.src.layers.object_detection.roi_generator import ROIGenerator
from keras_cv.src.layers.object_detection.roi_align import _ROIAligner
from keras_cv.src.bounding_box.utils import _clip_boxes
from keras_cv.src.bounding_box.converters import _decode_deltas_to_boxes

from keras_cv.src.models.task import Task
from keras_cv.src.utils.train import get_feature_extractor
from keras_cv.src.models.object_detection.faster_rcnn import FeaturePyramid
from keras_cv.src.models.object_detection.faster_rcnn import RPNHead
from keras_cv.src.models.object_detection.faster_rcnn import RCNNHead

BOX_VARIANCE = [0.1, 0.1, 0.2, 0.2]

@keras_cv_export(
    ["keras_cv.models.FasterRCNN", "keras_cv.models.object_detection.FasterRCNN"]
)
class FasterRCNN(Task):
    def __init__(
        self,
        batch_size,
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
        extractor_levels = ["P2", "P3", "P4", "P5"]
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
                bounding_box_format,
                
            )
        )
        
        # 4. RPN Head
        num_anchors_per_location = len(scales) * len(aspect_ratios)
        rpn_head = RPNHead(
            num_anchors_per_location
        )
        
        # 5. ROI Generator
        roi_generator = ROIGenerator(
            bounding_box_format=bounding_box_format,
            nms_score_threshold_train=float("-inf"),
            nms_score_threshold_test=float("-inf"),
            name="roi_generator",
        )
        
        # 6. ROI Pooler
        roi_pooler = _ROIAligner(bounding_box_format="yxyx", name="roi_pooler")
        
        
        # 7. RCNN Head
        rcnn_head = rcnn_head or RCNNHead(
            num_classes,
            name="rcnn_head"
        )
        
        # Begin construction of forward pass
        image_shape = feature_extractor.input_shape[1:]  # exclude the batch size
        images = keras.layers.Input(
            image_shape,
            name="images",
            batch_size=batch_size,
        )
        
        backbone_outputs = feature_extractor(images)
        feature_map = feature_pyramid(backbone_outputs)
        
        # [BS, num_anchors, 4], [BS, num_anchors, 1]
        rpn_boxes, rpn_scores = rpn_head(feature_map)
        
        # Generate Anchors
        if None in image_shape:
            raise ValueError("Input image shape not provided.")
        anchors = anchor_generator(image_shape=image_shape)
        
        decoded_rpn_boxes = _decode_deltas_to_boxes(
            anchors=anchors,
            boxes_delta=rpn_boxes,
            anchor_format=bounding_box_format,
            box_format=bounding_box_format,
            variance=BOX_VARIANCE,
        )
        rpn_box_pred = keras.ops.concatenate(tree.flatten(rpn_boxes), axis=1)
        rpn_cls_pred = keras.ops.concatenate(tree.flatten(rpn_scores), axis=1)
        
        # Generate ROI's from RPN head
        rois, _ = roi_generator(decoded_rpn_boxes, rpn_scores)
        rois = _clip_boxes(rois, bounding_box_format, image_shape)
        
        # Pool the region of interests
        feature_map = roi_pooler(features=feature_map, boxes=rois)
        
        # Reshape the feature map [BS, H*W*K]
        feature_map = keras.ops.reshape(
            feature_map,
            newshape=keras.ops.shape(rois)[:2] + (-1,),
        )
        
        # Pass final feature map to RCNN Head for predictions
        box_pred, cls_pred = rcnn_head(feature_map=feature_map)
        
        inputs = {"images": images}
        box_pred = keras.layers.Concatenate(axis=1, name="box")([box_pred])
        cls_pred = keras.layers.Concatenate(axis=1, name="classification")(
            [cls_pred]
        )
        rpn_box_pred = keras.layers.Concatenate(axis=1, name="rpn_box")(
            [rpn_box_pred]
        )
        rpn_cls_pred = keras.layers.Concatenate(
            axis=1, name="rpn_classification"
        )([rpn_cls_pred])
        outputs = {
            "box": box_pred,
            "classification": cls_pred,
            "rpn_box": rpn_box_pred,
            "rpn_classification": rpn_cls_pred,
        }
        
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )
        
    
    @staticmethod
    def default_anchor_generator(scales, aspect_ratios, bounding_box_format):
        strides={f"P{i}": 2**i for i in range(2, 7)}
        sizes = {
            "P2": 32.0,
            "P3": 64.0,
            "P4": 128.0,
            "P5": 256.0,
            "P6": 512.0,
        }
        return cv_layers.AnchorGenerator(
            bounding_box_format=bounding_box_format,
            sizes=sizes,
            aspect_ratios=aspect_ratios,
            scales=scales,
            strides=strides,
            clip_boxes=True,
            name="anchor_generator",
        )
        
        
        
        