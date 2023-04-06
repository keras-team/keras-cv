from keras_cv.models import YOLOv8_N
import tensorflow as tf

model = YOLOv8_N()

# Run prediction
from keras_cv_attention_models import test_images

imm = test_images.dog_cat()
for_pred = tf.image.resize(imm, (640, 640))
preds = model(tf.expand_dims(for_pred, axis=0))
bboxs, lables, confidences = model.decode(preds[0])

# Show result
from keras_cv_attention_models.coco import data

data.show_image_with_bboxes(imm, bboxs, lables, confidences)


from keras_cv.models.object_detection.yolo_v8.old import YOLOv8_O

old_model = YOLOv8_O(pretrained="coco")


model.layers[1].load_weights("backbone.h5")

# Conversion:
#
# model.layers[1].load_weights("backbone.h5")
# for index in range(6, 150):
#   model.layers[index].set_weights(old_model.layers[index+115].get_weights())
