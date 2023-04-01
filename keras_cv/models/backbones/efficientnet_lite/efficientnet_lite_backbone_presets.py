def EfficientNetLiteB0(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    **kwargs,
):
    return EfficientNetLite(
        include_rescaling,
        include_top,
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=224,
        dropout_rate=0.2,
        name="efficientnetliteb0",
        weights=parse_weights(weights, include_top, "efficientnetliteb0"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def EfficientNetLiteB1(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    **kwargs,
):
    return EfficientNetLite(
        include_rescaling,
        include_top,
        width_coefficient=1.0,
        depth_coefficient=1.1,
        default_size=240,
        dropout_rate=0.2,
        name="efficientnetliteb1",
        weights=parse_weights(weights, include_top, "efficientnetliteb1"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def EfficientNetLiteB2(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    **kwargs,
):
    return EfficientNetLite(
        include_rescaling,
        include_top,
        width_coefficient=1.1,
        depth_coefficient=1.2,
        default_size=260,
        dropout_rate=0.3,
        name="efficientnetliteb2",
        weights=parse_weights(weights, include_top, "efficientnetliteb2"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def EfficientNetLiteB3(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    **kwargs,
):
    return EfficientNetLite(
        include_rescaling,
        include_top,
        width_coefficient=1.2,
        depth_coefficient=1.4,
        default_size=280,
        dropout_rate=0.3,
        name="efficientnetliteb3",
        weights=parse_weights(weights, include_top, "efficientnetliteb3"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def EfficientNetLiteB4(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    **kwargs,
):
    return EfficientNetLite(
        include_rescaling,
        include_top,
        width_coefficient=1.4,
        depth_coefficient=1.8,
        default_size=300,
        dropout_rate=0.3,
        name="efficientnetliteb4",
        weights=parse_weights(weights, include_top, "efficientnetliteb4"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


EfficientNetLiteB0.__doc__ = BASE_DOCSTRING.format(name="EfficientNetLiteB0")
EfficientNetLiteB1.__doc__ = BASE_DOCSTRING.format(name="EfficientNetLiteB1")
EfficientNetLiteB2.__doc__ = BASE_DOCSTRING.format(name="EfficientNetLiteB2")
EfficientNetLiteB3.__doc__ = BASE_DOCSTRING.format(name="EfficientNetLiteB3")
EfficientNetLiteB4.__doc__ = BASE_DOCSTRING.format(name="EfficientNetLiteB4")
