# Preprocessing Layers

KerasCV offers a bunch of pre-processing layers.

The `call()` method supports two formats of inputs:

1. A single image tensor with shape (height, width, channels) or (batch_size, height, width, channels)
2. A dict of tensors with any of the following keys (note that `"images"` must be present):
    * `"images"` - Image Tensor with shape (height, width, channels) or (batch_size, height, width, channels)
    * `"labels"` - One-hot encoded classification labels Tensor with shape (num_classes) or (batch_size, num_classes)
    * `"bounding_boxes"` - A dictionary with keys:
        * `"boxes"` - Tensor with shape (num_boxes, 4) or (batch_size num_boxes, 4)
        * `"classes"` - Tensor of class labels for boxes with shape (num_boxes, num_classes) or (batch_size, num_boxes, num_classes).
    * `"segmentation_masks"` - Tensor with shape (height, width, num_classes) or (batch_size, height, width, num_classes)
    Any other keys included in this dictionary will be ignored and unmodified by an augmentation layer.

The provided table gives an overview of the different augmentation layers available and the data formats they support.

| Layer Name | Vectorized | Segmentation Masks | BBoxes | Class Labels |
| :-- | :--: | :--: | :--: | :--: |
| AugMix | ❌ | ❌ | ✅ | ✅ |
| AutoContrast | ✅ | ✅ | ✅ | ✅ |
| ChannelShuffle | ✅ | ✅ | ✅ | ✅ |
| CutMix | ❌ | ✅ | ❌ | ✅ |
| Equalization | ❌ | ✅ | ✅ | ✅ |
| FourierMix | ❌ | ❌ | ❌ | ✅ |
| Grayscale | ✅ | ✅ | ✅ | ✅ |
| GridMask | ❌ | ✅ | ✅ | ✅ |
| JitteredResize | ✅ | ✅ | ✅ | ✅ |
| MixUp | ❌ | ✅ | ✅ | ✅ |
| Mosaic | ✅ | ✅ | ✅ | ✅ |
| Posterization | ❌ | ✅ | ✅ | ✅ |
| RandAugment | ❌ | ❌ | ❌ | ❌ |
| RandomApply <sup>+</sup> | - | - | - | - |
| RandomAspectRatio | ❌ | ❌ | ✅ | ✅ |
| RandomBrightness | ✅| ✅ | ✅ | ✅ |
| RandomChannelShift | ❌| ✅ | ✅ | ✅ |
| RandomChoice <sup>+</sup> | - | - | - | - |
| RandomColorDegeneration | ❌ | ✅ | ✅ | ✅ |
| RandomColorJitter | ✅ | ✅ | ✅ | ✅ |
| RandomContrast | ✅ | ✅ | ✅ | ✅ |
| RandomCropAndResize | ❌ | ✅ | ✅ | ❌ |
| RandomCrop | ✅ | ❌ | ✅ | ✅ |
| RandomCutout | ❌ | ❌ | ❌ | ✅ |
| RandomFlip | ✅ | ✅ | ✅ | ✅ |
| RandomGaussianBlur | ❌ | ✅ | ✅ | ✅ |
| RandomHue | ✅ | ✅ | ✅ | ✅ |
| RandomJpegQuality | ❌ | ✅ | ✅ | ✅ |
| RandomRotation | ✅ | ✅ | ✅ | ✅ |
| RandomSaturation | ✅ | ✅ | ✅ | ✅ |
| RandomSharpness | ✅ | ✅ | ✅ | ✅ |
| RandomShear | ✅ | ❌ | ✅ | ✅ |
| RandomTranslation | ✅ | ❌ | ✅ | ✅ |
| RandomZoom | ✅ | ❌ | ❌ | ✅ |
| RepeatedAugmentation | ❌ | ❌ | ❌ | ❌ |
| Rescaling | ❌ | ✅ | ✅ | ✅ |
| Resizing | ❌ | ✅ | ✅ | ❌ |
| Solarization | ✅ | ✅ | ✅ | ✅ |

<sup>+</sup> Meta Layers, the data types will depend on the Sub Layers.

# Base Layers

- BaseImageAugmentationLayer
- VectorizedBaseImageAugmentationLayer
- RandomAugmentationPipeline