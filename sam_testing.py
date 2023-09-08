import keras_cv
from keras_cv.backend import keras
from keras_cv.backend import multi_backend
from keras_cv.models import TwoWayTransformer

if multi_backend():
    keras.config.disable_traceback_filtering()
else:
    import tensorflow as tf

    tf.debugging.disable_traceback_filtering()

image_encoder = keras_cv.models.ViTDetBBackbone()
prompt_encoder = keras_cv.models.SAMPromptEncoder(
    embed_dim=256,
    image_embedding_size=(64, 64),
    input_image_size=(1024, 1024),
    mask_in_chans=16,
)
mask_decoder = keras_cv.models.SAMMaskDecoder(
    transformer_dim=256,
    transformer=TwoWayTransformer(
        depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8
    ),
    num_multimask_outputs=3,
    iou_head_depth=3,
    iou_head_hidden_dim=256,
)

model = keras_cv.models.SegmentAnythingModel(
    image_encoder, prompt_encoder, mask_decoder
)
