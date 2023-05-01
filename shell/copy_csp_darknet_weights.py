import tensorflow as tf

from keras_cv.models import CSPDarkNetBackbone
from keras_cv.models import YOLOV8Backbone
from keras_cv.models.object_detection.yolo_v8.yolo_v8_backbone_presets import (
    copy_weights,
)

yolo_model = YOLOV8Backbone.from_preset("yolov8_xl_backbone_coco")

csp_model = CSPDarkNetBackbone.from_preset("yolov8_xl_backbone")
copy_weights(yolo_model, csp_model)

print(yolo_model(tf.ones(shape=(1, 512, 512, 3)))[0, 0, 0, :5])
print(csp_model(tf.ones(shape=(1, 512, 512, 3)))[0, 0, 0, :5])
