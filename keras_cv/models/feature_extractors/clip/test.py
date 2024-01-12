import numpy as np

from keras_cv.backend import keras
from keras_cv.models.feature_extractors.clip.clip_model import CLIP_B16
from keras_cv.models.feature_extractors.clip.clip_processor import CLIPProcessor

processor = CLIPProcessor(2)
text = processor.process_texts(["my name is Divya"])

model = CLIP_B16()
print("done instatiating model")
text_features = model.encode_text(text)
print("text features", text_features)
# image_features, text_features = model.encode_pair(image, text)
