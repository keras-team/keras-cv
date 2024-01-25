from PIL import Image

from keras_cv.models import CLIP
from keras_cv.models.feature_extractors.clip import CLIPProcessor

MODEL_CONFIGS = {
    "CLIP_B32": {
        "embed_dim": 512,
        "context_length": 77,
        "vocab_size": 49408,
        "transformer_width": 512,
        "transformer_heads": 8,
        "transformer_layers": 12,
        "vision_layers": 12,
        "vision_width": 768,
        "image_resolution": 224,
        "vision_patch_size": 32,
    },
}
embed_dim = MODEL_CONFIGS["CLIP_B32"]["embed_dim"]
context_length = MODEL_CONFIGS["CLIP_B32"]["context_length"]
vocab_size = MODEL_CONFIGS["CLIP_B32"]["vocab_size"]
transformer_width = MODEL_CONFIGS["CLIP_B32"]["transformer_width"]
transformer_heads = MODEL_CONFIGS["CLIP_B32"]["transformer_heads"]
transformer_layers = MODEL_CONFIGS["CLIP_B32"]["transformer_layers"]
vision_layers = MODEL_CONFIGS["CLIP_B32"]["vision_layers"]
vision_width = MODEL_CONFIGS["CLIP_B32"]["vision_width"]
vision_patch_size = MODEL_CONFIGS["CLIP_B32"]["vision_patch_size"]
image_resolution = MODEL_CONFIGS["CLIP_B32"]["image_resolution"]


model_32 = CLIP(
    embed_dim,
    image_resolution,
    vision_layers,
    vision_width,
    vision_patch_size,
    context_length,
    vocab_size,
    transformer_width,
    transformer_heads,
    transformer_layers,
)

image = Image.open("keras_cv/models/feature_extractors/clip/cat.jpg")
processor = CLIPProcessor(224)
image = processor.process_images(
    ["keras_cv/models/feature_extractors/clip/cat.jpg"]
)
text = processor.process_texts(
    ["photo of two cats", "a photo of a cat", "car and a dog"]
)
print(image.shape)
print(text.shape)
image_logits, text_logits = model_32(image, text)
print(image_logits)

print(
    model_32.get_layer("clip_encoder")
    .get_layer("residual_transformer_encoder")
    .resblocks.get_layer("residual_attention")
    .attn.weights
)
print(
    model_32.get_layer("residual_transformer_encoder")
    .resblocks.get_layer("residual_attention_12")
    .attn.weights
)
model_32.summary()
