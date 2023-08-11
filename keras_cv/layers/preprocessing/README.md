# Preprocessing Layers

KerasCV offers many preprocessing and data augmentation layers which support classification, object detection, and segmentation masks. When you use KerasCV augmentation layers to augment your training data, class labels, bounding boxes, and mask labels automatically get augmented alongside the image augmentations!

The provided table gives an overview of the different augmentation layers available and the data formats they support.

| Layer Name | Vectorized | Segmentation Masks | BBoxes | Class Labels |
| :-- | :--: | :--: | :--: | :--: |
| AugMix | ❌ | ✅ | ✅ | ✅ |
| AutoContrast | ✅ | ✅ | ✅ | ✅ |
| ChannelShuffle | ✅ | ✅ | ✅ | ✅ |
| CutMix | ❌ | ✅ | ❌ | ✅ |
| Equalization | ❌ | ✅ | ✅ | ✅ |
| FourierMix | ❌ | ✅ | ❌ | ✅ |
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
| RandomShear | ✅ | ✅ | ✅ | ✅ |
| RandomTranslation | ✅ | ❌ | ✅ | ✅ |
| RandomZoom | ✅ | ❌ | ❌ | ✅ |
| RepeatedAugmentation <sup>+</sup> | - | - | - | - |
| Rescaling | ❌ | ✅ | ✅ | ✅ |
| Resizing | ❌ | ✅ | ✅ | ❌ |
| Solarization | ✅ | ✅ | ✅ | ✅ |

<sup>+</sup> Meta Layers, the data types will depend on the Sub Layers.

# Base Layers

- BaseImageAugmentationLayer
- VectorizedBaseImageAugmentationLayer
- RandomAugmentationPipeline