from tensorflow.keras import backend
from tensorflow.keras import layers


def get_input_tensor(input_shape, input_tensor):
    if input_tensor is None:
        return layers.Input(shape=input_shape)
    if not backend.is_keras_tensor(input_tensor):
        return layers.Input(tensor=input_tensor, shape=input_shape)
    return input_tensor
