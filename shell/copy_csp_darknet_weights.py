import tensorflow as tf
from tensorflow import keras

from keras_cv.models import CSPDarkNetBackbone
from keras_cv.models import YOLOV8Backbone


def get_all_layers(model):
    layers = []
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            layers += get_all_layers(layer)
        elif not isinstance(layer, keras.layers.InputLayer):
            layers.append(layer)
    return layers


# tiny_csp_model = CSPDarkNetBackbone.from_preset("csp_darknet_tiny_imagenet")


# for layer in get_all_layers(tiny_csp_model):
#     print(layer.name)

yolo_model = YOLOV8Backbone.from_preset("yolov8_xl_backbone_coco")

# print(len(get_all_layers(yolo_model)))
# for layer in get_all_layers(yolo_model):
#     print(layer.name)

# print("*****")
csp_model = CSPDarkNetBackbone.from_preset("yolov8_xl_backbone")
# print(len(get_all_layers(csp_model)))

# for layer in get_all_layers(csp_model):
#     print(layer.name)

yolo_weights = {
    layer.name: layer.get_weights() for layer in get_all_layers(yolo_model)
}
for layer in get_all_layers(csp_model):
    if "split" in layer.name:
        continue
    layer.set_weights(yolo_weights[layer.name])

print(yolo_model(tf.ones(shape=(1, 512, 512, 3)))[0, 0, 0, :5])
print(csp_model(tf.ones(shape=(1, 512, 512, 3)))[0, 0, 0, :5])

yolo_layers = {layer.name: layer for layer in get_all_layers(yolo_model)}
for layer in get_all_layers(csp_model):
    if "split" in layer.name:
        continue
    assert layer.get_config() == yolo_layers[layer.name].get_config()
