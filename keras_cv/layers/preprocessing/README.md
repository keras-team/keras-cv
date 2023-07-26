# Base Layers

- BaseImageAugmentationLayer
- VectorizedBaseImageAugmentationLayer
- RandomAugmentationPipeline

# Preprocessing Layers

| Layer Name | Base Layer | Segmentation Masks | BBoxes |
| :-- | :-- | :--: | :--: |
| AugMix | BaseImageAugmentationLayer | ❌ | ✅ |
| AutoContrast | VectorizedBaseImageAugmentationLayer | ✅ | ✅ |
| ChannelShuffle | VectorizedBaseImageAugmentationLayer | ✅ | ✅ |
| (TODO) CutMix | BaseImageAugmentationLayer | ✅ | ❌ |
| Equalization | BaseImageAugmentationLayer | ✅ | ✅ |
| FourierMix | BaseImageAugmentationLayer | ❌ | ❌ |
| Grayscale | VectorizedBaseImageAugmentationLayer | ✅ | ✅ |
| GridMask | BaseImageAugmentationLayer | ✅ | ✅ |
| JitteredResize | VectorizedBaseImageAugmentationLayer | ✅ | ✅ |
| (TODO) MixUp | BaseImageAugmentationLayer | ✅ | ✅ |
| Mosaic | VectorizedBaseImageAugmentationLayer | ✅ | ✅ |
| Posterization | BaseImageAugmentationLayer | ✅ | ✅ |
| (TODO) RandAugment | RandomAugmentationPipeline | ❌ | ❌ |
| RandomApply | BaseImageAugmentationLayer | ❌ | ❌ |
| RandomAspectRatio | BaseImageAugmentationLayer | ❌ | ✅ |
| RandomBrightness | VectorizedBaseImageAugmentationLayer| ✅ | ✅ |
| RandomChannelShift | BaseImageAugmentationLayer| ✅ | ✅ |
| RandomChoice | BaseImageAugmentationLayer | ❌ | ❌ |
| RandomColorDegeneration | BaseImageAugmentationLayer | ✅ | ✅ |
| RandomColorJitter | VectorizedBaseImageAugmentationLayer | ✅ | ✅ |
| RandomContrast | VectorizedBaseImageAugmentationLayer | ✅ | ✅ |
| RandomCropAndResize | BaseImageAugmentationLayer | ✅ | ✅ |
| RandomCrop | VectorizedBaseImageAugmentationLayer | ❌ | ✅ |
| RandomCutout | BaseImageAugmentationLayer | ❌ | ❌ |
| RandomFlip | VectorizedBaseImageAugmentationLayer | ✅ | ✅ |
| RandomGaussianBlur | BaseImageAugmentationLayer | ✅ | ✅ |
| RandomHue | VectorizedBaseImageAugmentationLayer | ✅ | ✅ |
| RandomJpegQuality | BaseImageAugmentationLayer | ✅ | ✅ |
| RandomRotation | VectorizedBaseImageAugmentationLayer | ✅ | ✅ |
| RandomSaturation | VectorizedBaseImageAugmentationLayer | ✅ | ✅ |
| RandomSharpness | VectorizedBaseImageAugmentationLayer | ✅ | ✅ |
| RandomShear | VectorizedBaseImageAugmentationLayer | ❌ | ✅ |
| RandomTranslation | VectorizedBaseImageAugmentationLayer | ❌ | ✅ |
| RandomZoom | VectorizedBaseImageAugmentationLayer | ❌ | ❌ |
| RepeatedAugmentation | BaseImageAugmentationLayer | ❌ | ❌ |
| Rescaling | BaseImageAugmentationLayer | ✅ | ✅ |
| (TODO) Resizing | BaseImageAugmentationLayer | ✅ | ✅ |
| Solarization | VectorizedBaseImageAugmentationLayer | ✅ | ✅ |
