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
"""
Mask R-CNN model based on the Tensorflow model garden implementation.

As an experiment, Keras CV will try to wrap a thin layer on top of model garden
implementation, so that we could quickly provide models to end user without
re-implement/re-train the model end to end.

This class might serve as a temporary solution, and might change its interface
and details if needed in the future.
"""

import tensorflow as tf

try:
    import tensorflow_models as tfm
except ImportError:
    # Didn't have tensorflow model garden package installed, and the model can't
    # be instantiated.
    tfm = None


def _assert_tensorflow_model_installed():
    if tfm is None:
        raise ImportError(
            'Tensor model package is not installed in the '
            'environment. Please use `pip install tf-models-official` '
            'or `pip install tf-models-nightly` to install the required '
            'dependency package')


def mask_rcnn(
    classes,
    input_size,
    backbone,
    backbone_weights,
) -> tf.keras.Model:
    """Create a MaskRCNN model.

    Args:
        classes: int, the number of classes for the detection model. Note that
            the classes doesn't contain the background class, and the classes
            from the data should be represented by integers with range
            [0, classes).
        input_size: list or tuple of ints, the shape of the input for the model.
            Note that this doesn't include the batch dimension. A sample use case
            is like [512, 512, 3].
        backbone: a backbone network for the model. Can be a `tf.keras.Model`
           instance. The supported pre-defined backbone models are
           1: 'resnet', which is a Resnet50 model.
           2: 'spinenet', which is SpineNet49 model.
           3: 'mobilenet', which is a MobileNet v2 model.
        backbone_weights: pre-trained weights for the backbone model. The weights
           can be a list of tensors, which will be set to backbone model
           via `model.set_weights()`. A string file path is also supported, to
           read the weights from a folder of tensorflow checkpoints. Note that
           the weights and backbone network should be compatible between each
           other.
    """
    _assert_tensorflow_model_installed()

    # Common model configs that are shared by all parts of the network
    # TODO(scottzhu): Those value should be tunable, based on different
    # backbone/decoder/RPN/
    # This setting is for ResNet 50
    min_level = 2
    max_level = 6
    anchor_aspect_ratios = [0.5, 1.0, 2.0]
    anchor_num_scales = 1
    anchor_size = 8
    num_anchors_per_location = len(anchor_aspect_ratios) * anchor_num_scales

    # Common normalization/regularization configs that are shared by all parts of the
    # network.
    activation = 'relu'
    use_sync_bn = True
    norm_momentum = 0.997
    norm_epsilon = 0.0001
    kernel_regularizer = tf.keras.regularizers.l2(0.00002)

    # ================== Backbone =====================
    if isinstance(backbone, str):
        supported_premade_backbone = ['resnet', 'spinenet', 'mobilenet']
        if backbone not in supported_premade_backbone:
            raise ValueError(
                f'Supported premade backbones are: '
                '{supported_premade_backbone}, received "{backbone}"')
        input_spec = tf.keras.layers.InputSpec(shape=[None] + list(input_size))
        if backbone == 'resnet':
            backbone = tfm.vision.backbones.ResNet(
                model_id=50,
                input_specs=input_spec,
                se_ratio=0.0,   # From the default Resnet config
                activation=activation,
                use_sync_bn=use_sync_bn,
                norm_momentum=norm_momentum,
                norm_epsilon=norm_epsilon,
                kernel_regularizer=kernel_regularizer
            )
        else:
            # TODO(scottzhu): Add spinenet and mobilenet later
            pass
    else:
        # Make sure the backbone is a Keras model instance
        # TODO(scottzhu): Might need to do more assertion about the model
        if not isinstance(backbone, tf.keras.Model):
            raise ValueError('Backbone need to be a `tf.keras.Model`, '
                             f'received {backbone}')

    # load the backbone weights if provided.
    # TODO(scottzhu):

    # ================== Decoder/FPN =====================
    # TODO(scottzhu): Might want to expose the decoder to the method signature.
    decoder_num_filters = 256
    decoder = tfm.vision.decoders.FPN(
        input_specs=backbone.output_specs,  # Might need to check for custom backbones
        min_level=min_level,
        max_level=max_level,
        num_filters=decoder_num_filters,
        activation=activation,
        use_sync_bn=use_sync_bn,
        norm_momentum=norm_momentum,
        norm_epsilon=norm_epsilon,
        kernel_regularizer=kernel_regularizer
    )

    # ================== RPN =====================
    # TODO(scottzhu): Might want to expose the RPN to the method signature.
    rpn_num_convs = 1
    rpn_num_filters = 256
    rpn_head = tfm.vision.heads.RPNHead(
        min_level=min_level,
        max_level=max_level,
        num_anchors_per_location=num_anchors_per_location,
        num_convs=rpn_num_convs,
        num_filters=rpn_num_filters,
        activation=activation,
        use_sync_bn=use_sync_bn,
        norm_momentum=norm_momentum,
        norm_epsilon=norm_epsilon,
        kernel_regularizer=kernel_regularizer
    )

    # ================== Detection head =====================
    # TODO(scottzhu): Might want to expose the detection head to the method signature.
    detection_num_convs = 4
    detection_num_filters = 256
    detection_num_fcs = 1
    detection_fc_dims = 1024
    detection_class_agnostic_bbox_pred = False
    detection_head = tfm.vision.heads.DetectionHead(
        num_classes=classes,
        num_convs=detection_num_convs,
        num_filters=detection_num_filters,
        num_fcs=detection_num_fcs,
        fc_dims=detection_fc_dims,
        class_agnostic_bbox_pred=detection_class_agnostic_bbox_pred,
        activation=activation,
        use_sync_bn=use_sync_bn,
        norm_momentum=norm_momentum,
        norm_epsilon=norm_epsilon,
        kernel_regularizer=kernel_regularizer,
        name='detection_head')

    # Create feature output tensor from backbone and RPN
    backbone_features = backbone(tf.keras.Input(shape=input_size))
    decoder_features = decoder(backbone_features)
    rpn_head(decoder_features)

    # ================== ROI generator =====================
    roi_generator_pre_nms_top_k = 2000
    roi_generator_pre_nms_score_threshold = 0.0
    roi_generator_pre_nms_min_size_threshold = 0.0
    roi_generator_nms_iou_threshold = 0.7
    roi_generator_num_proposals = 1000
    roi_generator_test_pre_nms_top_k = 1000
    roi_generator_test_pre_nms_score_threshold = 0.0
    roi_generator_test_pre_nms_min_size_threshold = 0.0
    roi_generator_test_nms_iou_threshold = 0.7
    roi_generator_test_num_proposals = 1000
    roi_generator_use_batched_nms = False
    roi_generator = tfm.vision.layers.roi_generator.MultilevelROIGenerator(
        pre_nms_top_k=roi_generator_pre_nms_top_k,
        pre_nms_score_threshold=roi_generator_pre_nms_score_threshold,
        pre_nms_min_size_threshold=roi_generator_pre_nms_min_size_threshold,
        nms_iou_threshold=roi_generator_nms_iou_threshold,
        num_proposals=roi_generator_num_proposals,
        test_pre_nms_top_k=roi_generator_test_pre_nms_top_k,
        test_pre_nms_score_threshold=roi_generator_test_pre_nms_score_threshold,
        test_pre_nms_min_size_threshold=roi_generator_test_pre_nms_min_size_threshold,
        test_nms_iou_threshold=roi_generator_test_nms_iou_threshold,
        test_num_proposals=roi_generator_test_num_proposals,
        use_batched_nms=roi_generator_use_batched_nms
    )

    # ================== ROI =====================
    roi_sampler_mix_gt_boxes = True
    roi_sampler_num_sampled_rois = 512
    roi_sampler_foreground_fraction = 0.25
    roi_sampler_foreground_iou_threshold = 0.5
    roi_sampler_background_iou_high_threshold = 0.5
    roi_sampler_background_iou_low_threshold = 0.0
    roi_sampler = tfm.vision.layers.ROISampler(
        mix_gt_boxes=roi_sampler_mix_gt_boxes,
        num_sampled_rois=roi_sampler_num_sampled_rois,
        foreground_fraction=roi_sampler_foreground_fraction,
        foreground_iou_threshold=roi_sampler_foreground_iou_threshold,
        background_iou_high_threshold=roi_sampler_background_iou_high_threshold,
        background_iou_low_threshold=roi_sampler_background_iou_low_threshold
    )

    roi_aligner_crop_size = 7
    roi_aligner_sample_offset = 0.5
    roi_aligner = tfm.vision.layers.MultilevelROIAligner(
        crop_size=roi_aligner_crop_size,
        sample_offset=roi_aligner_sample_offset,
    )

    # ================= Detection Generator ======================
    detection_generator_apply_nms = True
    detection_generator_pre_nms_top_k = 5000
    detection_generator_pre_nms_score_threshold = 0.05
    detection_generator_nms_iou_threshold = 0.5
    detection_generator_max_num_detections = 100
    detection_generator_nms_version = 'v2'
    detection_generator_use_cpu_nms = False
    detection_generator_soft_nms_sigma = None
    detection_generator = tfm.vision.layers.DetectionGenerator(
        apply_nms=detection_generator_apply_nms,
        pre_nms_top_k=detection_generator_pre_nms_top_k,
        pre_nms_score_threshold=detection_generator_pre_nms_score_threshold,
        nms_iou_threshold=detection_generator_nms_iou_threshold,
        max_num_detections=detection_generator_max_num_detections,
        nms_version=detection_generator_nms_version,
        use_cpu_nms=detection_generator_use_cpu_nms,
        soft_nms_sigma=detection_generator_soft_nms_sigma,
    )

    # =============== Mask head and sampler =======================
    mask_head_upsample_factor = 2
    mask_head_num_convs = 4
    mask_head_num_filters = 256
    mask_head_use_separable_conv = False
    mask_head_class_agnostic = False
    mask_sampler_num_mask = 128
    mask_roi_aligner_crop_size = 14
    mask_roi_aligner_sample_offset = 0.5

    mask_head = tfm.vision.heads.MaskHead(
        num_classes=classes,
        upsample_factor=mask_head_upsample_factor,
        num_convs=mask_head_num_convs,
        num_filters=mask_head_num_filters,
        use_separable_conv=mask_head_use_separable_conv,
        activation=activation,
        use_sync_bn=use_sync_bn,
        norm_momentum=norm_momentum,
        norm_epsilon=norm_epsilon,
        kernel_regularizer=kernel_regularizer,
        class_agnostic=mask_head_class_agnostic,
    )
    mask_sampler = tfm.vision.layers.MaskSampler(
        mask_target_size=mask_roi_aligner_crop_size * mask_head_upsample_factor,
        num_sampled_masks=mask_sampler_num_mask,
    )
    mask_roi_aligner = tfm.vision.layers.MultilevelROIAligner(
        crop_size=mask_roi_aligner_crop_size,
        sample_offset=mask_roi_aligner_sample_offset,
    )

    mask_rcnn_model = tfm.vision.maskrcnn_model.MaskRCNNModel(
        backbone=backbone,
        decoder=decoder,
        rpn_head=rpn_head,
        detection_head=detection_head,
        roi_generator=roi_generator,
        roi_sampler=roi_sampler,
        roi_aligner=roi_aligner,
        detection_generator=detection_generator,
        mask_head=mask_head,
        mask_sampler=mask_sampler,
        mask_roi_aligner=mask_roi_aligner,
        class_agnostic_bbox_pred=detection_class_agnostic_bbox_pred,
        cascade_class_ensemble=False,
        min_level=min_level,
        max_level=max_level,
        num_scales=anchor_num_scales,
        aspect_ratios=anchor_aspect_ratios,
        anchor_size=anchor_size,
    )

    # TODO(scottzhu): rewrite the model.train/eval functions.
    # See http://google3/third_party/tensorflow_models/official/vision/tasks/maskrcnn.py;l=47;rcl=468079502
    return mask_rcnn_model
