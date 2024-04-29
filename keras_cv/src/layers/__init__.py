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

from tensorflow.keras.layers import CenterCrop
from tensorflow.keras.layers import RandomHeight
from tensorflow.keras.layers import RandomWidth

from keras_cv.src.layers.augmenter import Augmenter
from keras_cv.src.layers.feature_pyramid import FeaturePyramid
from keras_cv.src.layers.fusedmbconv import FusedMBConvBlock
from keras_cv.src.layers.hierarchical_transformer_encoder import (
    HierarchicalTransformerEncoder,
)
from keras_cv.src.layers.mbconv import MBConvBlock
from keras_cv.src.layers.object_detection.anchor_generator import (
    AnchorGenerator,
)
from keras_cv.src.layers.object_detection.box_matcher import BoxMatcher
from keras_cv.src.layers.object_detection.multi_class_non_max_suppression import (  # noqa: E501
    MultiClassNonMaxSuppression,
)
from keras_cv.src.layers.object_detection.non_max_suppression import (
    NonMaxSuppression,
)
from keras_cv.src.layers.object_detection_3d.centernet_label_encoder import (
    CenterNetLabelEncoder,
)
from keras_cv.src.layers.object_detection_3d.voxelization import (
    DynamicVoxelization,
)
from keras_cv.src.layers.overlapping_patching_embedding import (
    OverlappingPatchingAndEmbedding,
)
from keras_cv.src.layers.preprocessing.aug_mix import AugMix
from keras_cv.src.layers.preprocessing.auto_contrast import AutoContrast
from keras_cv.src.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.src.layers.preprocessing.channel_shuffle import ChannelShuffle
from keras_cv.src.layers.preprocessing.cut_mix import CutMix
from keras_cv.src.layers.preprocessing.equalization import Equalization
from keras_cv.src.layers.preprocessing.fourier_mix import FourierMix
from keras_cv.src.layers.preprocessing.grayscale import Grayscale
from keras_cv.src.layers.preprocessing.grid_mask import GridMask
from keras_cv.src.layers.preprocessing.jittered_resize import JitteredResize
from keras_cv.src.layers.preprocessing.mix_up import MixUp
from keras_cv.src.layers.preprocessing.mosaic import Mosaic
from keras_cv.src.layers.preprocessing.posterization import Posterization
from keras_cv.src.layers.preprocessing.rand_augment import RandAugment
from keras_cv.src.layers.preprocessing.random_apply import RandomApply
from keras_cv.src.layers.preprocessing.random_aspect_ratio import (
    RandomAspectRatio,
)
from keras_cv.src.layers.preprocessing.random_augmentation_pipeline import (
    RandomAugmentationPipeline,
)
from keras_cv.src.layers.preprocessing.random_brightness import RandomBrightness
from keras_cv.src.layers.preprocessing.random_channel_shift import (
    RandomChannelShift,
)
from keras_cv.src.layers.preprocessing.random_choice import RandomChoice
from keras_cv.src.layers.preprocessing.random_color_degeneration import (
    RandomColorDegeneration,
)
from keras_cv.src.layers.preprocessing.random_color_jitter import (
    RandomColorJitter,
)
from keras_cv.src.layers.preprocessing.random_contrast import RandomContrast
from keras_cv.src.layers.preprocessing.random_crop import RandomCrop
from keras_cv.src.layers.preprocessing.random_crop_and_resize import (
    RandomCropAndResize,
)
from keras_cv.src.layers.preprocessing.random_cutout import RandomCutout
from keras_cv.src.layers.preprocessing.random_flip import RandomFlip
from keras_cv.src.layers.preprocessing.random_gaussian_blur import (
    RandomGaussianBlur,
)
from keras_cv.src.layers.preprocessing.random_hue import RandomHue
from keras_cv.src.layers.preprocessing.random_jpeg_quality import (
    RandomJpegQuality,
)
from keras_cv.src.layers.preprocessing.random_rotation import RandomRotation
from keras_cv.src.layers.preprocessing.random_saturation import RandomSaturation
from keras_cv.src.layers.preprocessing.random_sharpness import RandomSharpness
from keras_cv.src.layers.preprocessing.random_shear import RandomShear
from keras_cv.src.layers.preprocessing.random_translation import (
    RandomTranslation,
)
from keras_cv.src.layers.preprocessing.random_zoom import RandomZoom
from keras_cv.src.layers.preprocessing.repeated_augmentation import (
    RepeatedAugmentation,
)
from keras_cv.src.layers.preprocessing.rescaling import Rescaling
from keras_cv.src.layers.preprocessing.resizing import Resizing
from keras_cv.src.layers.preprocessing.solarization import Solarization
from keras_cv.src.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    VectorizedBaseImageAugmentationLayer,
)
from keras_cv.src.layers.preprocessing_3d.waymo.frustum_random_dropping_points import (  # noqa: E501
    FrustumRandomDroppingPoints,
)
from keras_cv.src.layers.preprocessing_3d.waymo.frustum_random_point_feature_noise import (  # noqa: E501
    FrustumRandomPointFeatureNoise,
)
from keras_cv.src.layers.preprocessing_3d.waymo.global_random_dropping_points import (  # noqa: E501
    GlobalRandomDroppingPoints,
)
from keras_cv.src.layers.preprocessing_3d.waymo.global_random_flip import (
    GlobalRandomFlip,
)
from keras_cv.src.layers.preprocessing_3d.waymo.global_random_rotation import (
    GlobalRandomRotation,
)
from keras_cv.src.layers.preprocessing_3d.waymo.global_random_scaling import (
    GlobalRandomScaling,
)
from keras_cv.src.layers.preprocessing_3d.waymo.global_random_translation import (  # noqa: E501
    GlobalRandomTranslation,
)
from keras_cv.src.layers.preprocessing_3d.waymo.group_points_by_bounding_boxes import (  # noqa: E501
    GroupPointsByBoundingBoxes,
)
from keras_cv.src.layers.preprocessing_3d.waymo.random_copy_paste import (
    RandomCopyPaste,
)
from keras_cv.src.layers.preprocessing_3d.waymo.random_drop_box import (
    RandomDropBox,
)
from keras_cv.src.layers.preprocessing_3d.waymo.swap_background import (
    SwapBackground,
)
from keras_cv.src.layers.regularization.drop_path import DropPath
from keras_cv.src.layers.regularization.dropblock_2d import DropBlock2D
from keras_cv.src.layers.regularization.squeeze_excite import SqueezeAndExcite2D
from keras_cv.src.layers.regularization.stochastic_depth import StochasticDepth
from keras_cv.src.layers.segformer_multihead_attention import (
    SegFormerMultiheadAttention,
)
from keras_cv.src.layers.spatial_pyramid import SpatialPyramidPooling
from keras_cv.src.layers.transformer_encoder import TransformerEncoder
from keras_cv.src.layers.vit_det_layers import AddRelativePositionalEmbedding
from keras_cv.src.layers.vit_det_layers import MultiHeadAttentionWithRelativePE
from keras_cv.src.layers.vit_det_layers import ViTDetPatchingAndEmbedding
from keras_cv.src.layers.vit_det_layers import WindowedTransformerEncoder
from keras_cv.src.layers.vit_det_layers import WindowPartitioning
from keras_cv.src.layers.vit_layers import PatchingAndEmbedding
